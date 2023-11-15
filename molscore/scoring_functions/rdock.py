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
    
    @staticmethod
    def add_pharmacophore_constraint(config, constraints=None, optional_constraints=None, n_optional=1):
        if constraints and optional_constraints:
             config += \
f"""
#################################
## PHARMACOPHORIC RESTRAINTS
#################################
SECTION PHARMA
    SCORING_FUNCTION RbtPharmaSF
    WEIGHT 1.0
    CONSTRAINTS_FILE {os.path.basename(constraints)}
    OPTIONAL_FILE {os.path.basename(optional_constraints)}
    NOPT {n_optional}
END_SECTION
"""
        elif optional_constraints and not constraints:
            config += \
f"""
#################################
## PHARMACOPHORIC RESTRAINTS
#################################
SECTION PHARMA
    SCORING_FUNCTION RbtPharmaSF
    WEIGHT 1.0
    OPTIONAL_FILE {os.path.basename(optional_constraints)}
    NOPT {n_optional}
END_SECTION
"""
        else:
            config += \
f"""
#################################
## PHARMACOPHORIC RESTRAINTS
#################################
SECTION PHARMA
    SCORING_FUNCTION RbtPharmaSF
    WEIGHT 1.0
    CONSTRAINTS_FILE {os.path.basename(constraints)}
END_SECTION
"""
        return config

    def __init__(self, prefix: str, receptor: Union[str, os.PathLike], ref_ligand: Union[str, os.PathLike],
                 cluster: Union[str, int] = None, 
                 ligand_preparation: str = 'GypsumDL', prep_timeout: float = 30.0,
                 dock_protocol: Union[str, os.PathLike] = 'dock', dock_timeout: float = 120.0, n_runs: int = 5,
                 dock_constraints: Union[str, os.PathLike] = None,
                 dock_opt_constraints: Union[str, os.PathLike] = None, dock_n_opt_constraints: int = 1,
                 **kwargs):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param receptor: Path to receptor file (.pdb)
        :param ref_ligand: Path to ligand file for autobox generation (.sdf, .pdb)
        :param cluster: Address to Dask scheduler for parallel processing via dask or number of local workers to use
        :param ligand_preparation: Use LigPrep (default), rdkit stereoenum + Epik most probable state, Moka+Corina abundancy > 20 or GypsumDL [LigPrep, Epik, Moka, GypsumDL]
        :param prep_timeout: Timeout (seconds) before killing a ligand preparation process (e.g., long running RDKit jobs)
        :param dock_protocol: Select from docking protocols or path to a custom .prm protocol [dock, dock_solv, dock_grid, dock_solv_grid, minimise, minimise_solv, score, score_solv]
        :param dock_timeout: Timeout (seconds) before killing an individual docking simulation
        :param n_runs: Number of docking trials (poses to return)
        :param dock_constrains: Path to rDock pharmacophoric constriants file that are mandatory
        :param dock_opt_constraints: Path to rDock pharmacophoric constriants file that are optional
        :param dock_n_opt_constraints: Number of optional constraints required
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
        self.n_runs = n_runs
        self.rdock_env = 'rbdock'
        self.rcav_env = 'rbcavity'
        self.dock_constraints = os.path.abspath(dock_constraints) if dock_constraints else dock_constraints
        self.dock_opt_constraints = os.path.abspath(dock_opt_constraints) if dock_opt_constraints else dock_opt_constraints
        self.dock_n_opt_constraints = dock_n_opt_constraints
        self.rdock_files = [self.receptor, self.ref]

        # Setup dask
        self.cluster = cluster
        self.client = DaskUtils.setup_dask(
            cluster_address_or_n_workers=self.cluster,
            local_directory=self.temp_dir.name,
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
        if dock_protocol in ['dock', 'dock_solv', 'dock_grid', 'dock_solv_grid', 'minimise', 'minimise_solv', 'score', 'score_solv']:
            self.dock_protocol = dock_protocol + ".prm"
        else:
            self.dock_protocol = os.path.abspath(dock_protocol)
        
        # Prepare grid file in tempfiles
        os.environ['RBT_HOME'] = self.temp_dir.name
        # Write config
        rec_prm = self.cavity_config(self.receptor, self.ref)
        # Add PH4 constriants
        if self.dock_constraints or self.dock_opt_constraints:
            rec_prm = self.add_pharmacophore_constraint(rec_prm, self.dock_constraints, self.dock_opt_constraints, self.dock_n_opt_constraints)
            if self.dock_constraints: self.rdock_files.append(self.dock_constraints)
            if self.dock_opt_constraints: self.rdock_files.append(self.dock_opt_constraints)
        # Copy files to RBT home
        self.subprocess.run(f'cp {" ".join(self.rdock_files)} {self.temp_dir.name}')
        # Write config
        self.rec_prm = os.path.join(self.temp_dir.name, 'cavity.prm')
        with open(self.rec_prm, 'wt') as f:
            f.write(rec_prm)
        self.rdock_files.append(self.rec_prm)
        # Prepare cavity
        self.subprocess.run(f"{self.rcav_env} -was -r {self.rec_prm}")
        self.cavity = os.path.join(self.temp_dir.name, 'cavity.as')
        self.rdock_files.append(self.cavity)

    def _close_dask(self):
        if self.client:
            self.client.close()
            # If local cluster close that too, can't close remote cluster
            try: self.client.cluster.close()
            except: pass
    
    def _move_rdock_files(self, cwd):
        os.environ['RBT_HOME'] = cwd
        self.subprocess.run(f'cp {" ".join(self.rdock_files)} {cwd}')
    
    def reformat_ligands(self, varients, varient_files):
        """Reformat prepared ligands to .sd via obabel (RDKit doesn't write charge in a rDock friendly way)"""
        futures = []
        new_varient_files = []
        for vfile in varient_files:
            new_vfile = vfile.replace(".sdf", ".sd")
            new_varient_files.append(new_vfile)
            if self.cluster:
                p = timedSubprocess()
                futures.append(self.client.submit(p.run, f"obabel {vfile} -O {new_vfile}"))
            else:
                self.subprocess.run(f"obabel -i {vfile} -O {new_vfile}")
        
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
                command = f'{self.rdock_env} -i {in_lig} -o {out_lig} -r cavity.prm -p {self.dock_protocol} -n {self.n_runs} --allH'
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
                        rdock_out = Chem.ForwardSDMolSupplier(out_file, sanitize=False) 
                        for mol in rdock_out:  
                            mol = manually_charge_mol(mol) # RDKit doesn't like rDock charged files
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

    def __call__(self, smiles: list, directory: str, file_names: list, cleanup: bool = True, **kwargs):
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
        if cleanup: self.remove_files(keep=best_variants, parallel=True)
        fh.close()
        logger.removeHandler(fh)
        self.directory = None
        self.file_names = None
        self.variants = None

        # Check
        assert len(smiles) == len(self.docking_results)

        return self.docking_results
    
def manually_charge_mol(mol):
    """Manually charge non-full valent atoms, assuming all explicit hydrogens are present"""
    PT = Chem.GetPeriodicTable()
    for atom in mol.GetAtoms():
        e = atom.GetSymbol()
        # Skip sulfur
        if e == 'S':
            continue
        v = atom.GetExplicitValence()
        dv = PT.GetDefaultValence(atom.GetAtomicNum())
        if (v < dv) and (atom.GetFormalCharge() == 0):
            atom.SetFormalCharge(-1)
        if (v > dv) and (atom.GetFormalCharge() == 0):
            atom.SetFormalCharge(1)
    Chem.SanitizeMol(mol)
    return Chem.RemoveHs(mol)