"""
Makes use of and adapted from Gypsum-DL
    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0358-3
    https://durrantlab.pitt.edu/gypsum-dl/
As well as pyscreener,
    https://github.com/coleygroup/pyscreener
"""

import os
import logging
import glob
from itertools import takewhile
from dask.distributed import Client

from molscore.utils.gypsum_dl.Parallelizer import Parallelizer
from molscore.utils.gypsum_dl.Start import prepare_smiles, prepare_3d, add_mol_id_props
from molscore.utils.gypsum_dl.MolContainer import MolContainer

from rdkit import Chem

from molscore.scoring_functions.utils import timedSubprocess

logger = logging.getLogger('smina')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class SminaDock:
    """
    Score structures based on their Smina docking score, using Gypsum-DL for ligand preparation
    """
    return_metrics = ['docking_score']

    def __init__(self, prefix: str, receptor: os.PathLike, ref_ligand: os.PathLike, cpus: int = 1,
                 cluster: str = None, timeout: float = 120.0):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param receptor: Path to receptor file (.pdbqt)
        :param ref_ligand: Path to ligand file for autobox generation (.sdf/.pdb)
        :param cpus: Number of Smina CPUs to use per simulation
        :param cluster: Address to Dask scheduler for parallel processing via dask
        :param timeout: Timeout (seconds) before killing an individual docking simulation
        """
        self.prefix = prefix.replace(" ", "_")
        self.receptor = os.path.abspath(receptor)
        self.ref = os.path.abspath(ref_ligand)
        self.file_names = None
        self.variants = None
        self.cpus = cpus
        self.cluster = cluster
        if self.cluster is not None:
            self.client = Client(self.cluster)
        self.timeout = timeout

    def preprocess_ligands(self, smiles):
        """
        Prepare ligands and save to sdf files <file_name_gypsum.sdf>
        :param smiles:
        :return: file locations
        """
        # Set Gypsum-DL parameters
        gypsum_params = {
            "source": "",
            "output_folder": "./",
            "separate_output_files": True,
            "add_pdb_output": False,
            "add_html_output": False,
            "num_processors": -1,
            "start_time": 0,
            "end_time": 0,
            "run_time": 0,
            "min_ph": 7.0,
            "max_ph": 7.0,
            "pka_precision": 1.0,
            "thoroughness": 1,
            "max_variants_per_compound": 8,
            "second_embed": False,
            "2d_output_only": False,
            "skip_optimize_geometry": True,
            "skip_alternate_ring_conformations": True,
            "skip_adding_hydrogen": False,
            "skip_making_tautomers": False,
            "skip_enumerate_chiral_mol": False,
            "skip_enumerate_double_bonds": False,
            "let_tautomers_change_chirality": False,
            "use_durrant_lab_filters": True,
            "job_manager": "multiprocessing",
            "cache_prerun": False,
            "test": False,
        }
        gypsum_params["Parallelizer"] = Parallelizer(
            gypsum_params["job_manager"], gypsum_params["num_processors"], True
        )

        # Load SMILES data
        smiles_data = [(smi, name, {}) for smi, name in zip(smiles, self.file_names)]

        # Make the molecule containers.
        contnrs = []
        for i in range(0, len(smiles_data)):
            smiles, name, props = smiles_data[i]
            new_contnr = MolContainer(smiles, name, i, props)
            contnrs.append(new_contnr)

        # Remove None types from failed conversion
        contnrs = [x for x in contnrs if x.orig_smi_canonical != None]

        # Prepare and embed
        prepare_smiles(contnrs, gypsum_params)

        # Convert the processed SMILES strings to 3D.
        prepare_3d(contnrs, gypsum_params)

        # Add in name and unique id to each molecule.
        add_mol_id_props(contnrs)

        # Save the output.
        output_paths = []
        self.variants = {name: [] for name in self.file_names}
        for i, contnr in enumerate(contnrs):
            # First of all remove duplicates
            contnr.remove_identical_mols_from_contnr()
            for v, m in enumerate(contnr.mols):
                if m:
                    self.variants[contnr.name].append(v)
                    m.load_conformers_into_rdkit_mol()
                    path = os.path.join(self.directory, f'{contnr.name}-{v}_gypsum.sdf')
                    w = Chem.SDWriter(path)
                    w.write(m.rdkit_mol)
                    w.flush()
                    w.close()
                    output_paths.append(path)
                    logger.debug(f'Written {contnr.name}-{v}')
        return output_paths

    def dock_ligands(self, ligand_paths):
        smina_commands = []
        log_paths = []
        for l in ligand_paths:
            out_file = os.path.join(self.directory, os.path.basename(l).replace("_gypsum.sdf", "_docked.sdf"))
            out_log = os.path.join(self.directory, os.path.basename(l).replace("_gypsum.sdf", "_log.txt"))
            log_paths.append(out_log)
            cmd = f"smina -r {self.receptor} -l {l} --autobox_ligand {self.ref} -o {out_file} " \
                  f"--cpu {self.cpus} --exhaustiveness 8 --energy_range 3 --min_rmsd_filter 1 --quiet " \
                  f"--log {out_log}"
            smina_commands.append(cmd)

        # Initialize subprocess
        logger.debug('Smina called')
        p = timedSubprocess(timeout=self.timeout).run

        if self.cluster is not None:
            futures = self.client.map(p, smina_commands)
            _ = self.client.gather(futures)
        else:
            _ = [p(command) for command in smina_commands]
        logger.debug('Smina finished')
        return log_paths

    @staticmethod
    def parse_log_file(log_file: str):
        # vina-type log files have scoring information between this
        # table border and the line: "Writing output ... done."
        TABLE_BORDER = '-----+------------+----------+----------'
        try:
            with open(log_file) as fid:
                for line in fid:
                    if TABLE_BORDER in line:
                        break

                score_lines = takewhile(
                    lambda line: 'Writing' not in line, fid)
                scores = [float(line.split()[1])
                          for line in score_lines]

            if len(scores) == 0:
                score = None
            else:
                scores = sorted(scores)
                score = scores[0]  # Take best score
        except OSError:
            score = None

        return score

    def get_docking_scores(self, smiles: list, return_best_variant: bool = False):
        """
        Read output sdfs, get output properties
        :param smiles: List of SMILES strings
        :param return_best_variant:
        :return optional, list of filenames with best variant
        """
        # Read in docked file
        best_variants = self.file_names.copy()
        best_score = {name: None for name in self.file_names}

        # For each molecule
        for i, (smi, name) in enumerate(zip(smiles, self.file_names)):
            docking_result = {'smiles': smi}

            # For each variant
            for variant in self.variants[name]:
                try:
                    log_file = os.path.join(self.directory, f'{name}-{variant}_log.txt')
                    dscore = self.parse_log_file(log_file)
                    if dscore is not None:
                        # If molecule doesn't have a score yet append it and the variant
                        if best_score[name] is None:
                            best_score[name] = dscore
                            best_variants[i] = f'{name}-{variant}'
                            docking_result.update({f'{self.prefix}_docking_score': dscore})
                            logger.debug(f'Docking score for {name}-{variant}: {dscore}')
                        # If docking score is better change it...
                        elif dscore < best_score[name]:
                            best_score[name] = dscore
                            best_variants[i] = f'{name}-{variant}'
                            docking_result.update({f'{self.prefix}_docking_score': dscore})
                            logger.debug(f'Found better {name}-{variant}: {dscore}')
                        # Otherwise ignore
                        else:
                            pass
                    # If path doesn't exist and nothing store, append 0
                    else:
                        logger.debug(f'{name}-{variant}_log.txt does not exist')
                        if best_score[name] is None:  # Only if no other score for prefix
                            best_variants[i] = f'{name}-{variant}'
                            docking_result.update({f'{self.prefix}_docking_score': 0.0})
                            logger.debug(f'Returning 0.0 unless a successful variant is found')
                # If parsing the molecule threw an error and nothing stored, append 0
                except:
                    logger.debug(f'Error processing {name}-{variant}_log.txt')
                    if best_score[name] is None:  # Only if no other score for prefix
                        best_variants[i] = f'{name}-{variant}'
                        docking_result.update({f'{self.prefix}_docking_score': 0.0})
                        logger.debug(f'Returning 0.0 unless a successful variant is found')

            # Add best variant information to docking result
            docking_result.update({f'{self.prefix}_best_variant': best_variants[i]})
            self.docking_results.append(docking_result)

        logger.debug(f'Best scores: {best_score}')
        if return_best_variant:
            logger.debug(f'Returning best variants: {best_variants}')
            return best_variants

        return self

    def remove_files(self, keep: list = [], parallel: bool = True):
        """
        Remove some of the log files and molecule files.
        :param keep: List of filenames to keep pose files for.
        :param parallel: Whether to run using Dask (requires scheduler address during initialisation).
        """
        # If no cluster is provided ensure parallel is False
        if (parallel is True) and (self.cluster is None):
            parallel = False

        keep_poses = [f'{k}_docked.sdf' for k in keep]
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
        self.directory = os.path.join(os.path.abspath(directory), 'SminaDock', step)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        # Add logging file handler
        fh = logging.FileHandler(os.path.join(self.directory, f'{step}_log.txt'))
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Prepare ligands
        output_paths = self.preprocess_ligands(smiles=smiles)

        # Dock ligands
        self.dock_ligands(output_paths)

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
