import atexit
import glob
import logging
import os
import subprocess
import warnings
from itertools import cycle, takewhile
from tempfile import TemporaryDirectory
from typing import Union

from rdkit import Chem

from molscore.scoring_functions._ligand_preparation import ligand_preparation_protocols
from molscore.scoring_functions.descriptors import MolecularDescriptors
from molscore.scoring_functions.utils import DaskUtils, check_openbabel, timedSubprocess

logger = logging.getLogger("gnina")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class GninaDock:
    """
    Score structures based on their Gnina docking score, using Gypsum-DL for ligand preparation
    """

    return_metrics = [
        "docking_score",
        "CNN_pose",
        "CNN_affinity",
        "NetCharge",
        "PositiveCharge",
        "NegativeCharge",
        "best_variant",
    ]

    @staticmethod
    def check_installation():
        if "gnina" not in os.environ.keys():
            raise RuntimeError(
                "Can't find gnina in PATH, please export the gnina executable as 'gnina', an executable file can be downloaded here https://github.com/gnina/gnina/releases/"
            )

    def __init__(
        self,
        prefix: str,
        receptor: Union[str, os.PathLike],
        ref_ligand: Union[str, os.PathLike],
        cpus: int = 1,
        cluster: Union[str, int] = None,
        ligand_preparation: str = "GypsumDL",
        prep_timeout: float = 30.0,
        dock_timeout: float = 180.0,
        gpu_device: int = -1,
        **kwargs,
    ):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param receptor: Path to receptor file (.pdb, .pdbqt)
        :param ref_ligand: Path to ligand file for autobox generation (.sdf, .pdb)
        :param cpus: Number of Smina CPUs to use per simulation
        :param cluster: Address to Dask scheduler for parallel processing via dask or number of local workers to use
        :param ligand_preparation: Use LigPrep (default), rdkit stereoenum + Epik most probable state, Moka+Corina abundancy > 20 or GypsumDL [LigPrep, Epik, Moka, GypsumDL]
        :param prep_timeout: Timeout (seconds) before killing a ligand preparation process (e.g., long running RDKit jobs)
        :param dock_timeout: Timeout (seconds) before killing an individual docking simulation
        :param gpu_device: Which GPU device to use for CNN scoring, -1 uses all available GPUs on machine
        """
        # Check requirements
        self.check_installation()
        check_openbabel()

        # If receptor is pdb, convert
        if receptor.endswith(".pdb"):
            pdbqt_receptor = receptor.replace(".pdb", ".pdbqt")
            subprocess.run(["obabel", receptor, "-O", pdbqt_receptor])
            receptor = pdbqt_receptor

        # Specify class attributes
        self.prefix = prefix.replace(" ", "_")
        self.receptor = os.path.abspath(receptor)
        self.ref = os.path.abspath(ref_ligand)
        self.file_names = None
        self.variants = None
        self.cpus = cpus
        self.dock_timeout = float(dock_timeout)
        if "timeout" in kwargs.items():
            self.dock_timeout = float(kwargs["timeout"])  # Back compatability
        self.prep_timeout = float(prep_timeout)
        self.temp_dir = TemporaryDirectory()
        self.gnina_env = os.environ["gnina"]

        # Set GPU's to use, one model takes about 1.5GB, for simple parallelisation, smarter would be to seperate docking and CNN scoring
        self.gpu = gpu_device
        if self.gpu:
            from torch import cuda

            self.ndevice = cuda.device_count()
            if gpu_device >= self.ndevice:
                warnings.warn(
                    f"GPU index selected {gpu_device} is more than the number of devices recognised by cuda {self.ndevice}",
                    ResourceWarning,
                )
            # Incase cuda isn't recognising gpus
            if (self.gpu == -1) and (self.ndevice == 0):
                self.ndevice = 1

        # Setup dask
        self.client = DaskUtils.setup_dask(
            cluster_address_or_n_workers=cluster,
            local_directory=self.temp_dir.name,
            logger=logger,
        )
        atexit.register(DaskUtils._close_dask, self.client)

        # Select ligand preparation protocol
        self.ligand_protocol = [
            p
            for p in ligand_preparation_protocols
            if ligand_preparation.lower() == p.__name__.lower()
        ][0]  # Back compatible
        if self.client is not None:
            self.ligand_protocol = self.ligand_protocol(
                dask_client=self.client, timeout=self.prep_timeout, logger=logger
            )
        else:
            self.ligand_protocol = self.ligand_protocol(logger=logger)

    def dock_ligands(self, ligand_paths):
        gnina_commands = []
        log_paths = []
        if self.gpu == -1:
            devices = cycle(range(0, self.ndevice))
        for lpath in ligand_paths:
            out_file = os.path.join(
                self.directory,
                os.path.basename(lpath).replace("_prepared.sdf", "_docked.sdf"),
            )
            out_log = os.path.join(
                self.directory,
                os.path.basename(lpath).replace("_prepared.sdf", "_log.txt"),
            )
            log_paths.append(out_log)
            if self.gpu:
                if self.gpu == -1:
                    d = next(devices)
                    device = f"--device {d}"
                else:
                    device = f"--device {self.gpu}"
            else:
                device = "--no_gpu"
            cmd = (
                f"{self.gnina_env} -r {self.receptor} -l {lpath} --autobox_ligand {self.ref} -o {out_file} "
                f"--cpu {self.cpus} --quiet "
                f"--log {out_log} {device}"
            )
            gnina_commands.append(cmd)

        # Initialize subprocess
        logger.debug("Gnina called")
        p = timedSubprocess(timeout=self.dock_timeout).run

        if self.client is not None:
            futures = self.client.map(p, gnina_commands)
            results = self.client.gather(futures)
        else:
            results = [p(command) for command in gnina_commands]
        logger.debug("Gnina finished")
        _ = [logger.warning(err.decode()) for out, err in results if err != "".encode()]
        return log_paths

    @staticmethod
    def parse_log_file(log_file: str):
        # vina-type log files have scoring information between this
        # table border and the line: "Writing output ... done."
        TABLE_BORDER = "-----+------------+------------+----------"
        try:
            with open(log_file) as fid:
                for line in fid:
                    if TABLE_BORDER in line:
                        break

                score_lines = takewhile(lambda line: "Writing" not in line, fid)
                scores = [line.split()[1:] for line in score_lines]

            if len(scores) == 0:
                score = None
            else:
                scores = sorted(scores, key=lambda x: float(x[0]))
                score = scores[0]  # Take best score
                score = [float(s) for s in score]
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
            docking_result = {"smiles": smi}

            # For each variant
            for variant in self.variants[name]:
                try:
                    # Get best score from log file
                    log_file = os.path.join(self.directory, f"{name}-{variant}_log.txt")
                    dscores = self.parse_log_file(log_file)
                    # Get associated Mol
                    mol_file = os.path.join(
                        self.directory, f"{name}-{variant}_docked.sdf"
                    )
                    gnina_out = Chem.ForwardSDMolSupplier(mol_file, sanitize=False)
                    mol = next(gnina_out)  # Mol file is ordered by dscore
                    if dscores is not None:
                        # If molecule doesn't have a score yet append it and the variant
                        if best_score[name] is None:
                            # Add charge info
                            net_charge, positive_charge, negative_charge = (
                                MolecularDescriptors.charge_counts(mol)
                            )
                            best_score[name] = dscores[2]
                            best_variants[i] = f"{name}-{variant}"
                            docking_result.update(
                                {
                                    f"{self.prefix}_docking_score": dscores[0],
                                    f"{self.prefix}_CNN_pose": dscores[1],
                                    f"{self.prefix}_CNN_affinity": dscores[2],
                                    f"{self.prefix}_NetCharge": net_charge,
                                    f"{self.prefix}_PositiveCharge": positive_charge,
                                    f"{self.prefix}_NegativeCharge": negative_charge,
                                }
                            )
                            logger.debug(
                                f"Docking score for {name}-{variant}: {dscores[2]}"
                            )
                        # If docking score is better change it...
                        elif dscores[2] < best_score[name]:
                            # Add charge info
                            net_charge, positive_charge, negative_charge = (
                                MolecularDescriptors.charge_counts(mol)
                            )
                            best_score[name] = dscores[2]
                            best_variants[i] = f"{name}-{variant}"
                            docking_result.update(
                                {
                                    f"{self.prefix}_docking_score": dscores[0],
                                    f"{self.prefix}_CNN_pose": dscores[1],
                                    f"{self.prefix}_CNN_affinity": dscores[2],
                                    f"{self.prefix}_NetCharge": net_charge,
                                    f"{self.prefix}_PositiveCharge": positive_charge,
                                    f"{self.prefix}_NegativeCharge": negative_charge,
                                }
                            )
                            logger.debug(f"Found better {name}-{variant}: {dscores[2]}")
                        # Otherwise ignore
                        else:
                            pass
                    # If path doesn't exist and nothing store, append 0
                    else:
                        logger.debug(f"{name}-{variant}_log.txt does not exist")
                        if (
                            best_score[name] is None
                        ):  # Only if no other score for prefix
                            best_variants[i] = f"{name}-{variant}"
                            docking_result.update(
                                {
                                    f"{self.prefix}_" + k: 0.0
                                    for k in self.return_metrics
                                }
                            )
                            logger.debug(
                                "Returning 0.0 unless a successful variant is found"
                            )
                # If parsing the molecule threw an error and nothing stored, append 0
                except Exception:
                    logger.debug(f"Error processing {name}-{variant}_log.txt")
                    if best_score[name] is None:  # Only if no other score for prefix
                        best_variants[i] = f"{name}-{variant}"
                        docking_result.update(
                            {f"{self.prefix}_" + k: 0.0 for k in self.return_metrics}
                        )
                        logger.debug(
                            "Returning 0.0 unless a successful variant is found"
                        )

            # Add best variant information to docking result
            docking_result.update({f"{self.prefix}_best_variant": best_variants[i]})
            self.docking_results.append(docking_result)

        logger.debug(f"Best scores: {best_score}")
        if return_best_variant:
            logger.debug(f"Returning best variants: {best_variants}")
            return best_variants

        return self

    def remove_files(self, keep: list = [], parallel: bool = True):
        """
        Remove some of the log files and molecule files.
        :param keep: List of filenames to keep pose files for.
        :param parallel: Whether to run using Dask (requires scheduler address during initialisation).
        """
        # If no cluster is provided ensure parallel is False
        if (parallel is True) and (self.client is None):
            parallel = False

        keep_poses = [f"{k}_docked.sdf" for k in keep]
        logger.debug(f"Keeping pose files: {keep_poses}")
        del_files = []
        for name in self.file_names:
            # Grab files
            files = glob.glob(os.path.join(self.directory, f"{name}*"))
            logger.debug(f"Glob found {len(files)} files")

            if len(files) > 0:
                try:
                    files = [
                        file
                        for file in files
                        if "log.txt" not in file
                        and not any([p in file for p in keep_poses])
                    ]

                    if parallel:
                        [del_files.append(file) for file in files]
                    else:
                        [os.remove(file) for file in files]
                # No need to stop if files can't be found and deleted
                except FileNotFoundError:
                    logger.debug("File not found.")
                    pass

        if parallel:
            futures = self.client.map(os.remove, del_files)
            _ = self.client.gather(futures)
        return self

    def __call__(
        self,
        smiles: list,
        directory: str,
        file_names: list,
        cleanup: bool = True,
        **kwargs,
    ):
        # Assign some attributes
        step = file_names[0].split("_")[0]  # Assume first Prefix is step
        self.file_names = file_names
        self.docking_results = []

        # Create log directory
        self.directory = os.path.join(
            os.path.abspath(directory), f"{self.prefix}_GninaDock", step
        )
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        # Add logging file handler
        fh = logging.FileHandler(os.path.join(self.directory, f"{step}_log.txt"))
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Prepare ligands
        self.variants, variant_paths = self.ligand_protocol(
            smiles=smiles, directory=self.directory, file_names=self.file_names
        )

        # Dock ligands
        self.dock_ligands(variant_paths)

        # Process output
        best_variants = self.get_docking_scores(smiles, return_best_variant=True)

        # Cleanup
        if cleanup:
            self.remove_files(keep=best_variants, parallel=True)
        fh.close()
        logger.removeHandler(fh)
        self.directory = None
        self.file_names = None
        self.variants = None

        # Check
        assert len(smiles) == len(self.docking_results)

        return self.docking_results
