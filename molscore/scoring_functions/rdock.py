import os
import atexit
import logging
import glob
import shutil
import subprocess
from functools import partial
from typing import Union
from itertools import takewhile
from tempfile import TemporaryDirectory

from rdkit import Chem

from molscore.scoring_functions.utils import timedSubprocess, DaskUtils
from molscore.scoring_functions.descriptors import MolecularDescriptors
from molscore.scoring_functions._ligand_preparation import ligand_preparation_protocols

logger = logging.getLogger('rdock')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class rDock:
    """
    Scores structures based on their rDock docking score
    """
    return_metrics = [
        'SCORE', 
        'SCORE.INTER',
        'SCORE.INTRA',
        'NetCharge', 'PositiveCharge', 'NegativeCharge',
        'best_variant'
        ]

    @staticmethod
    def check_installation():
        return shutil.which('rbdock')

    @staticmethod
    def cavity_config(receptor_file, ligand_file):
        config = \
f"""RBT_PARAMETER_FILE_V1.00
TITLE gart_DUD

RECEPTOR_FILE {os.path.basename(receptor_file)}
RECEPTOR_FLEX 3.0

##################################################################
### CAVITY DEFINITION: REFERENCE LIGAND METHOD
##################################################################
SECTION MAPPER
    SITE_MAPPER RbtLigandSiteMapper
    REF_MOL {os.path.basename(ligand_file)}
    RADIUS 8.0
    SMALL_SPHERE 1.0
    MIN_VOLUME 100
    MAX_CAVITIES 1
    VOL_INCR 0.0
GRIDSTEP 0.5
END_SECTION

#################################
#CAVITY RESTRAINT PENALTY
#################################
SECTION CAVITY
    SCORING_FUNCTION RbtCavityGridSF
    WEIGHT 1.0
END_SECTION
"""
        return config

    def __init__(self, prefix: str, receptor: Union[str, os.PathLike], ref_ligand: Union[str, os.PathLike],
                 cluster: Union[str, int] = None, 
                 dock_timeout: float = 120.0,
                 ligand_preparation: str = 'GypsumDL', prep_timeout: float = 30.0,
                 docking_protocol: Union[str, os.PathLike] = 'dock',  **kwargs):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param receptor: Path to receptor file (.pdb, .pdbqt)
        :param ref_ligand: Path to ligand file for autobox generation (.sdf, .pdb)
        :param cluster: Address to Dask scheduler for parallel processing via dask or number of local workers to use
        :param dock_timeout: Timeout (seconds) before killing an individual docking simulation
        :param ligand_preparation: Use LigPrep (default), rdkit stereoenum + Epik most probable state, Moka+Corina abundancy > 20 or GypsumDL [LigPrep, Epik, Moka, GypsumDL]
        :param prep_timeout: Timeout (seconds) before killing an ligand preparation process (e.g., long running RDKit jobs)
        :param docking_protocol: Select from docking protocols or path to a custom .prm protocol [dock, dock_solv, dock_grid, dock_solv_grid, minimise, minimise_solv, score, score_solv]
        """
        # Check rDock installation
        assert self.check_installation() is not None, "Could not find rDock path, please ensure proper installation"

        # Control subprocess in/out
        self.subprocess = timedSubprocess(timeout=None, shell=True)
        
        # If receptor is pdb, convert
        if receptor.endswith('.pdb'):
            mol2_receptor = receptor.replace('.pdb', '.mol2')
            self.subprocess.run(f"obabel {receptor} -O {mol2_receptor}")
            receptor = mol2_receptor

        # If ref_ligand doesn't end with sd, replace
        if not ref_ligand.endswith('.sd'):
            sd_ligand = ref_ligand.rsplit(".", 1)[0] + ".sd"
            self.subprocess.run(f"obabel {ref_ligand} -O {sd_ligand}")
            ref_ligand = sd_ligand
        
        # Specify class attributes
        self.prefix = prefix.replace(" ", "_")
        self.receptor = os.path.abspath(receptor)
        self.ref = os.path.abspath(ref_ligand)
        self.file_names = None
        self.variants = None
        self.dock_timeout = float(dock_timeout)
        if 'timeout' in kwargs.items(): self.dock_timeout = float(kwargs['timeout']) # Back compatability
        self.prep_timeout = float(prep_timeout)
        self.temp_dir = TemporaryDirectory()
        self.n_runs = 1
        self.rdock_env = 'rbdock'
        self.rcav_env = 'rbcavity'

        # Setup dask
        self.cluster = cluster
        if ligand_preparation == 'GypsumDL':
            processes = False
            if self.cluster: assert isinstance(cluster, (int, float)), "Must run local cluster to run GypsumDL"
        else:
            processes = True
        self.client = DaskUtils.setup_dask(
            cluster_address_or_n_workers=self.cluster,
            local_directory=self.temp_dir.name, 
            processes=processes,
            logger=logger
            )
        if self.client is None: self.cluster = None
        atexit.register(self._close_dask)

        # Select ligand preparation protocol
        self.ligand_protocol = [p for p in ligand_preparation_protocols if ligand_preparation.lower() == p.__name__.lower()][0] # Back compatible
        if self.cluster is not None:
            self.ligand_protocol = self.ligand_protocol(dask_client=self.client, timeout=self.prep_timeout, logger=logger)
        else:
            self.ligand_protocol = self.ligand_protocol(logger=logger)

        # Select docking protocol
        if docking_protocol in ['dock', 'dock_solv', 'dock_grid', 'dock_solv_grid', 'minimise', 'minimise_solv', 'score', 'score_solv']:
            self.docking_protocol = docking_protocol + ".prm"
        else:
            self.docking_protocol = os.path.abspath(docking_protocol)
        
        # Prepare grid file in tempfiles
        os.environ['RBT_HOME'] = self.temp_dir.name
        self.subprocess.run(f'cp {self.receptor} {self.ref} {self.temp_dir.name}')
        self.receptor_prm = os.path.join(self.temp_dir.name, 'cavity.prm')
        with open(self.receptor_prm, 'wt') as f:
            f.write(self.cavity_config(self.receptor, self.ref))
        self.subprocess.run(f"{self.rcav_env} -was -r {self.receptor_prm}")
        self.cavity = os.path.join(self.temp_dir.name, 'cavity.as')

    def _close_dask(self):
        if self.client:
            self.client.close()
    
    def _move_rdock_files(self, cwd):
        os.environ['RBT_HOME'] = cwd
        subprocess.run(['cp', self.receptor, self.ref, self.receptor_prm, self.cavity, cwd])
    
    def reformat_ligands(self, varients, varient_files):
        """Reformat prepared ligands to .mol2"""
        futures = []
        new_varient_files = []
        for vfile in varient_files:
            new_vfile = vfile.replace(".sdf", ".sd")
            new_varient_files.append(new_vfile)
            if self.cluster:
                futures.append(self.client.submit(shutil.move, vfile, new_vfile))
            else:
                shutil.move(vfile, new_vfile)
        
        # Wait for parallel jobs
        if self.cluster: self.client.gather(futures)
        
        return varients, new_varient_files

    def run_rDock(self, ligand_paths):
        # Move input files and set env
        self._move_rdock_files(self.directory)
        # Prepare rDock commands
        rdock_commands = []
        for name in self.file_names:
            for variant in self.variants[name]:
                in_lig = os.path.join(self.directory, f'{name}-{variant}_prepared.sd')
                out_lig = os.path.join(self.directory, f'{name}-{variant}_docked')
                out_log = os.path.join(self.directory, f'{name}-{variant}_rbdock.log')
                command = f'{self.rdock_env} -i {in_lig} -o {out_lig} -r cavity.prm -p dock.prm -n {self.n_runs} --allH'
                rdock_commands.append(command)

        # Initialize subprocess
        logger.debug('rDock called')
        p = timedSubprocess(timeout=self.dock_timeout)
        p = partial(p.run, cwd=self.directory)

        # Submit docking subprocesses
        if self.cluster is not None:
            futures = self.client.map(p, rdock_commands)
            results = self.client.gather(futures)
        else:
            results = [p(command) for command in rdock_commands]
        logger.debug('rDock finished')
        return self
    
    def get_docking_scores(self, smiles, return_best_variant=True):
        # Iterate over variants
        best_variants = self.file_names.copy()
        best_score = {name: None for name in self.file_names}

        # For each molecule
        for i, (smi, name) in enumerate(zip(smiles, self.file_names)):
            docking_result = {'smiles': smi}
            
            # If no variants enumerate 0
            if len(self.variants[name]) == 0:
                logger.debug(f'{name}_docked.sd does not exist')
                if best_score[name] is None:  # Only if no other score for prefix
                    docking_result.update({f'{self.prefix}_' + k: 0.0 for k in self.return_metrics})
                    logger.debug(f'Returning 0.0 as no variants exist')

            # For each variant
            for variant in self.variants[name]:
                out_file = os.path.join(self.directory, f'{name}-{variant}_docked.sd')
                if os.path.exists(out_file):
                    # Try to load it in, and grab the score
                    try:
                        rdock_out = Chem.ForwardSDMolSupplier(out_file)
                        for mol in rdock_out:  # should just be one
                            dscore = mol.GetPropsAsDict()['SCORE']
                            
                            # If molecule doesn't have a score yet append it and the variant
                            if best_score[name] is None:
                                best_score[name] = dscore
                                best_variants[i] = f'{name}-{variant}'
                                docking_result.update({f'{self.prefix}_' + k: v
                                                        for k, v in mol.GetPropsAsDict().items()
                                                        if k in self.return_metrics})
                                # Add charge info
                                net_charge, positive_charge, negative_charge = MolecularDescriptors.charge_counts(mol)
                                docking_result.update({f'{self.prefix}_NetCharge': net_charge,
                                                        f'{self.prefix}_PositiveCharge': positive_charge,
                                                        f'{self.prefix}_NegativeCharge': negative_charge})
                                logger.debug(f'Docking score for {name}-{variant}: {dscore}')

                            # If docking score is better change it...
                            elif dscore < best_score[name]:
                                best_score[name] = dscore
                                best_variants[i] = f'{name}-{variant}'
                                docking_result.update({f'{self.prefix}_' + k: v
                                                        for k, v in mol.GetPropsAsDict().items()
                                                        if k in self.return_metrics})
                                # Add charge info
                                net_charge, positive_charge, negative_charge = MolecularDescriptors.charge_counts(mol)
                                docking_result.update({f'{self.prefix}_NetCharge': net_charge,
                                                        f'{self.prefix}_PositiveCharge': positive_charge,
                                                        f'{self.prefix}_NegativeCharge': negative_charge})
                                logger.debug(f'Found better {name}-{variant}: {dscore}')

                            # Otherwise ignore
                            else:
                                pass

                    # If parsing the molecule threw an error and nothing stored, append 0
                    except:
                        logger.debug(f'Error processing {name}-{variant}_docked.sd file')
                        if best_score[name] is None:  # Only if no other score for prefix
                            best_variants[i] = f'{name}-{variant}'
                            docking_result.update({f'{self.prefix}_' + k: 0.0 for k in self.return_metrics})
                            logger.debug(f'Returning 0.0 unless a successful variant is found')

                # If path doesn't exist and nothing store, append 0
                else:
                    logger.debug(f'{name}-{variant}_docked.sd does not exist')
                    if best_score[name] is None:  # Only if no other score for prefix
                        best_variants[i] = f'{name}-{variant}'
                        docking_result.update({f'{self.prefix}_' + k: 0.0 for k in self.return_metrics})
                        logger.debug(f'Returning 0.0 unless a successful variant is found')

            # Add best variant information to docking result
            docking_result.update({f'{self.prefix}_best_variant': best_variants[i]})
            self.docking_results.append(docking_result)

        logger.debug(f'Best scores: {best_score}')
        if return_best_variant:
            logger.debug(f'Returning best variants: {best_variants}')
            return best_variants

    def remove_files(self, keep: list = [], parallel: bool = True):
        """
        Remove some of the log files and molecule files.
        :param keep: List of filenames to keep pose files for.
        :param parallel: Whether to run using Dask (requires scheduler address during initialisation).
        """
        # If no cluster is provided ensure parallel is False
        if (parallel is True) and (self.cluster is None):
            parallel = False

        keep_poses = [f'{k}_docked.sd' for k in keep]
        logger.debug(f'Keeping pose files: {keep_poses}')
        del_files = []
        for name in self.file_names:
            # Grab files
            files = glob.glob(os.path.join(self.directory, f'{name}*'))
            logger.debug(f'Glob found {len(files)} files')

            if len(files) > 0:
                try:
                    files = [file for file in files
                             if not ("log.txt" in file) and not any([p in file for p in keep_poses])]

                    if parallel:
                        [del_files.append(file) for file in files]
                    else:
                        [os.remove(file) for file in files]
                # No need to stop if files can't be found and deleted
                except FileNotFoundError:
                    logger.debug('File not found.')
                    pass

        if parallel:
            futures = self.client.map(os.remove, del_files)
            _ = self.client.gather(futures)
        return self

    def __call__(self, smiles: list, directory: str, file_names: list):
        # Assign some attributes
        step = file_names[0].split("_")[0]  # Assume first Prefix is step
        self.file_names = file_names
        self.docking_results = []

        # Create log directory
        self.directory = os.path.join(os.path.abspath(directory), 'rDock', step)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        # Add logging file handler
        fh = logging.FileHandler(os.path.join(self.directory, f'{step}_log.txt'))
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Prepare ligands
        self.variants, variant_paths = self.ligand_protocol(smiles=smiles, directory=self.directory, file_names=self.file_names, logger=logger)
        self.variants, variant_paths = self.reformat_ligands(self.variants, variant_paths)

        # Dock ligands
        self.run_rDock(variant_paths)

        # Process output
        best_variants = self.get_docking_scores(smiles, return_best_variant=True)

        # Cleanup
        self.remove_files(keep=best_variants, parallel=True)
        fh.close()
        logger.removeHandler(fh)
        self.directory = None
        self.file_names = None
        self.variants = None

        # Check
        assert len(smiles) == len(self.docking_results)

        return self.docking_results