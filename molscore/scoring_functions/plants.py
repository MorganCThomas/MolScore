import atexit
import glob
import logging
import os
from tempfile import TemporaryDirectory
from typing import Union

import pandas as pd
from rdkit.Chem import AllChem as Chem

from molscore.scoring_functions._ligand_preparation import ligand_preparation_protocols
from molscore.scoring_functions.descriptors import MolecularDescriptors
from molscore.scoring_functions.utils import DaskUtils, check_openbabel, timedSubprocess

logger = logging.getLogger("plants")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class PLANTSDock:
    """
    Score structures using PLANTS docking software
    Korb, O.; Stützle, T.; Exner, T. E. "PLANTS: Application of Ant Colony Optimization to Structure-Based Drug Design" Lecture Notes in Computer Science 4150, 247-258 (2006).
    http://www.tcd.uni-konstanz.de/plants_download/
    """

    return_metrics = [
        "TOTAL_SCORE",
        "SCORE_RB_PEN",
        "SCORE_NORM_HEVATOMS",
        "SCORE_NORM_CRT_HEVATOMS",
        "SCORE_NORM_WEIGHT",
        "SCORE_NORM_CRT_WEIGHT",
        "SCORE_RB_PEN_NORM_CRT_HEVATOMS",
        "SCORE_NORM_CONTACT",
        "EVAL",
        "TIME",
        "NetCharge",
        "PositiveCharge",
        "NegativeCharge",
        "best_variant",
    ]

    @staticmethod
    def check_installation():
        # Check installation
        if "PLANTS" not in list(os.environ.keys()):
            raise RuntimeError(
                "PLANTS installation not found, please install and add to os environment (e.g., export PLANTS=<path_to_plants_executable>)"
            )

    def __init__(
        self,
        prefix: str,
        receptor: Union[str, os.PathLike],
        ref_ligand: Union[str, os.PathLike],
        cluster: Union[str, int] = None,
        ligand_preparation: str = "GypsumDL",
        prep_timeout: float = 30.0,
        dock_timeout: float = 120.0,
        **kwargs,
    ):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param receptor: Protein receptor (.pdb, .mol2)
        :param ref_ligand: Reference ligand for identifying binding site (.sdf, .pdb, .mol2)
        :param cluster: Address to Dask scheduler for parallel processing via dask or number of local workers to use
        :param ligand_preparation: Use LigPrep (default), rdkit stereoenum + Epik most probable state, Moka+Corina abundancy > 20 or GypsumDL [LigPrep, Epik, Moka, GypsumDL]
        :param prep_timeout: Timeout (seconds) before killing a ligand preparation process (e.g., long running RDKit jobs)
        :param dock_timeout: Timeout before killing an individual docking simulation (seconds) (only if using Dask for parallelisation)
        :param kwargs:
        """
        # Check requirements
        check_openbabel()
        self.check_installation()

        # Convert necessary file formats
        self.subprocess = timedSubprocess(shell=True)
        if receptor.endswith(".pdb"):
            mol2_receptor = receptor.replace(".pdb", ".mol2")
            self.subprocess.run(
                cmd=f"obabel {receptor} -O {mol2_receptor} --partialcharge none"
            )
            receptor = mol2_receptor

        if ref_ligand.endswith(".pdb") or ref_ligand.endswith(".sdf"):
            ext = "." + ref_ligand.split(".")[-1]
            mol2_ligand = ref_ligand.replace(ext, ".mol2")
            self.subprocess.run(
                cmd=f"obabel {ref_ligand} -O {mol2_ligand} --partialcharge none"
            )
            ref_ligand = mol2_ligand

        # Specify class attributes
        self.prefix = prefix.replace(" ", "_")
        self.plants_metrics = self.return_metrics
        self.plants_env = os.environ["PLANTS"]
        self.dock_timeout = float(dock_timeout)
        if "timeout" in kwargs.items():
            self.dock_timeout = float(kwargs["timeout"])  # Back compatability
        self.prep_timeout = float(prep_timeout)
        self.receptor = os.path.abspath(receptor)
        self.ref_ligand = os.path.abspath(ref_ligand)
        self.temp_dir = TemporaryDirectory()
        self.variants = None
        self.docking_results = None

        # Setup dask
        self.client = DaskUtils.setup_dask(
            cluster_address_or_n_workers=cluster,
            local_directory=self.temp_dir.name,
            logger=logger,
        )
        atexit.register(DaskUtils._close_dask, self.client)

        # Set default PLANTS params & find binding site center
        self.params = {
            "scoring_function": "chemplp",
            "search_speed": "speed1",
            "write_multi_mol2": 1,
            "cluster_structures": 1,
            "cluster_rmsd": 2,
            "bindingsite_radius": 12,
            "protein_file": receptor,
        }
        # Find binding site
        self.subprocess.run(
            cmd=f"cd {self.temp_dir.name} ; {self.plants_env} --mode bind {self.ref_ligand} {self.params['bindingsite_radius']} {self.receptor}"
        )
        with open(os.path.join(self.temp_dir.name, "bindingsite.def"), "r") as f:
            for line in f.readlines():
                bs_args = line.strip("\n").split(" ")
                if bs_args[0] == "bindingsite_center":
                    self.params["bindingsite_center"] = " ".join(
                        [str(x) for x in bs_args[1:]]
                    )
                    break

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

    def reformat_ligands(self, varients, varient_files):
        """Reformat prepared ligands to .mol2"""
        new_varient_files = []
        for vfile in varient_files:
            new_vfile = vfile.replace(".sdf", ".mol2")
            self.subprocess.run(cmd=f"obabel {vfile} -O {new_vfile}")
            new_varient_files.append(new_vfile)
        return varients, new_varient_files

    def run_plants(self):
        """
        Write new input files and submit each to PLANTS
        """
        plants_commands = []
        for name in self.file_names:
            for variant in self.variants[name]:
                # Write config file
                params = self.params.copy()
                params["ligand_file"] = os.path.join(
                    self.directory, f"{name}-{variant}_prepared.mol2"
                )
                params["output_dir"] = os.path.join(self.directory, f"{name}-{variant}")
                config_file = os.path.join(self.directory, f"{name}-{variant}_config")
                with open(config_file, "wt") as f:
                    f.write("\n".join([f"{k} {v}" for k, v in params.items()]))

                # Run PLANTS with  {plants_env} --mode screen {config}
                command = f"{self.plants_env} --mode screen {config_file}"
                plants_commands.append(command)

        # Initialize subprocess
        logger.debug("PLANTS called")
        p = timedSubprocess(timeout=self.dock_timeout).run

        if self.client is not None:
            futures = self.client.map(p, plants_commands)
            results = self.client.gather(futures)
        else:
            results = [p(command) for command in plants_commands]
        logger.debug("PLANTS finished")
        _ = [logger.warning(err.decode()) for out, err in results if err != "".encode()]
        return self

    def get_docking_scores(self, smiles: list, return_best_variant: bool = False):
        """
        Extract docking scores from the results
        """
        # Read in docked file
        best_variants = self.file_names.copy()
        best_score = {name: None for name in self.file_names}

        # For each molecule
        for i, (smi, name) in enumerate(zip(smiles, self.file_names)):
            docking_result = {"smiles": smi}

            # If no variants ... next for loop won't run
            if len(self.variants[name]) == 0:
                logger.debug(f"{name}: No variants prepared")
                docking_result.update(
                    {f"{self.prefix}_" + k: 0.0 for k in self.return_metrics}
                )

            # For each variant
            for variant in self.variants[name]:
                out_file = os.path.join(
                    self.directory, f"{name}-{variant}", "ranking.csv"
                )
                if os.path.exists(out_file):
                    try:
                        # Try to load it in, and grab the first line score
                        plants_out = pd.read_csv(out_file).to_dict("records")[0]
                        plants_out.pop("LIGAND_ENTRY")
                        dscore = plants_out["TOTAL_SCORE"]

                        # If molecule doesn't have a score yet append it and the variant
                        if (best_score[name] is None) or (dscore < best_score[name]):
                            best_score[name] = dscore
                            best_variants[i] = f"{name}-{variant}"
                            docking_result.update(
                                {
                                    f"{self.prefix}_" + k: v
                                    for k, v in plants_out.items()
                                    if k in self.return_metrics
                                }
                            )
                            logger.debug(f"Best score for {name}-{variant}: {dscore}")
                            # Add charge info
                            try:
                                mol_file = os.path.join(
                                    self.directory,
                                    f"{name}-{variant}",
                                    "docked_ligands.mol2",
                                )
                                self.subprocess.run(
                                    f'obabel {mol_file} -O {mol_file.replace(".mol2", ".sdf")}'
                                )
                                mol_file = mol_file.replace(".mol2", ".sdf")
                                mol = next(Chem.SDMolSupplier(mol_file))
                                net_charge, positive_charge, negative_charge = (
                                    MolecularDescriptors.charge_counts(mol)
                                )
                                docking_result.update(
                                    {
                                        f"{self.prefix}_NetCharge": net_charge,
                                        f"{self.prefix}_PositiveCharge": positive_charge,
                                        f"{self.prefix}_NegativeCharge": negative_charge,
                                    }
                                )
                            except Exception as e:
                                docking_result.update(
                                    {
                                        f"{self.prefix}_" + k: 0.0
                                        for k in [
                                            "NetCharge",
                                            "PositiveCharge",
                                            "NegativeCharge",
                                        ]
                                    }
                                )
                                logger.debug(
                                    f"Error calculating charge for {name}-{variant}: {e}"
                                )

                    except Exception as e:
                        logger.debug(f"Error processing {name}-{variant} files: {e}")
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

                # If path doesn't exist and nothing store, append 0
                else:
                    logger.debug(f"{name}-{variant} output files do not exist")
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

    def remove_files(self, keep=[], parallel=True):
        """
        Remove some of the log files and molecule files.
        :param keep: List of filenames to keep pose files for.
        :param parallel: Whether to run using Dask
        """
        # If no cluster is provided ensure parallel is False
        if (parallel is True) and (self.client is None):
            parallel = False

        keep_files = [os.path.join(k, "docked_ligands.sdf") for k in keep] + [
            os.path.join(k, "ranking.csv") for k in keep
        ]
        keep_dirs = keep
        logger.debug(f"Keeping pose files: {keep_files}")
        del_files = []
        del_dirs = []
        for name in self.file_names:
            # Grab files
            files = glob.glob(os.path.join(self.directory, f"{name}*"))
            files += glob.glob(os.path.join(self.directory, f"{name}*", "*"))
            logger.debug(f"Glob found {len(files)} files")

            if len(files) > 0:
                try:
                    files = [
                        file
                        for file in files
                        if "log.txt" not in file
                        and not any([p in file for p in keep_files])
                    ]
                    dirs = [
                        d
                        for d in files
                        if os.path.isdir(d) and not any([p in d for p in keep_dirs])
                    ]

                    if parallel:
                        [
                            del_files.append(file)
                            for file in files
                            if not os.path.isdir(file)
                        ]
                        [del_dirs.append(d) for d in dirs]
                    else:
                        [os.remove(file) for file in files if not os.path.isdir(file)]
                        [os.rmdir(d) for d in dirs]
                # No need to stop if files can't be found and deleted
                except FileNotFoundError:
                    logger.debug("File not found.")
                    pass

        if parallel:
            futures = self.client.map(os.remove, del_files)
            _ = self.client.gather(futures)
            futures = self.client.map(os.rmdir, del_dirs)
            _ = self.client.gather(futures)
        return self

    def __call__(self, smiles: list, directory: str, file_names: list, **kwargs):
        """
        Calculate scores for GlideDock
        :param smiles: List of SMILES strings
        :param directory: Directory to save files and logs into
        :param file_names: List of corresponding file names for SMILES to match files to index
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        # Assign some attributes
        step = file_names[0].split("_")[0]  # Assume first Prefix is step

        # Create log directory
        self.directory = os.path.join(
            os.path.abspath(directory), f"{self.prefix}_PLANTSDock", step
        )
        os.makedirs(self.directory, exist_ok=True)
        self.file_names = file_names
        self.docking_results = []  # make sure no carry over

        # Add logging file handler
        fh = logging.FileHandler(os.path.join(self.directory, f"{step}_log.txt"))
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Refresh Dask every few hundred iterations
        if self.client is not None:
            if int(step) % 250 == 0:
                self.client.restart()

        # Run protocol
        self.variants, variant_files = self.ligand_protocol(
            smiles=smiles, directory=self.directory, file_names=file_names
        )
        self.variants, variant_files = self.reformat_ligands(
            self.variants, variant_files
        )
        self.run_plants()
        best_variants = self.get_docking_scores(smiles=smiles, return_best_variant=True)

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
