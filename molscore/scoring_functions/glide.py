import atexit
import glob
import gzip
import logging
import os
from tempfile import TemporaryDirectory
from typing import Union

from rdkit.Chem import AllChem as Chem

from molscore.scoring_functions._ligand_preparation import ligand_preparation_protocols
from molscore.scoring_functions.descriptors import MolecularDescriptors
from molscore.scoring_functions.utils import DaskUtils, timedSubprocess

logger = logging.getLogger("glide")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class GlideDock:
    """
    Score structures using Glide docking score, including ligand preparation with LigPrep
    """

    return_metrics = [
        "r_i_docking_score",
        "r_i_glide_ligand_efficiency",
        "r_i_glide_ligand_efficiency_sa",
        "r_i_glide_ligand_efficiency_ln",
        "r_i_glide_gscore",
        "r_i_glide_lipo",
        "r_i_glide_hbond",
        "r_i_glide_metal",
        "r_i_glide_rewards",
        "r_i_glide_evdw",
        "r_i_glide_ecoul",
        "r_i_glide_erotb",
        "r_i_glide_esite",
        "r_i_glide_emodel",
        "r_i_glide_energy",
        "NetCharge",
        "PositiveCharge",
        "NegativeCharge",
        "best_variant",
    ]

    def __init__(
        self,
        prefix: str,
        glide_template: Union[os.PathLike, str],
        cluster: Union[str, int] = None,
        ligand_preparation: str = "LigPrep",
        prep_timeout: float = 30.0,
        dock_timeout: float = 120.0,
        **kwargs,
    ):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param glide_template: Path to a template docking file (.in)
        :param cluster: Address to Dask scheduler for parallel processing via dask or number of local workers to use
        :param ligand_preparation: Use LigPrep (default), rdkit stereoenum + Epik most probable state, Moka+Corina abundancy > 20 or GypsumDL [LigPrep, Epik, Moka, GysumDL]
        :param prep_timeout: Timeout (seconds) before killing a ligand preparation process (e.g., long running RDKit jobs)
        :param dock_timeout: Timeout (seconds) before killing an individual docking simulation (only if using Dask for parallelisation)
        :param kwargs:
        """
        # Read in glide template (.in)
        with open(glide_template, "r") as gfile:
            self.glide_options = gfile.readlines()
        # Make sure output file type is sdf
        self.glide_options = self.modify_glide_in(
            self.glide_options, "POSE_OUTTYPE", "ligandlib_sd"
        )

        # Specify class attributes
        self.prefix = prefix.replace(" ", "_")
        self.glide_metrics = GlideDock.return_metrics
        self.glide_env = os.path.join(os.environ["SCHRODINGER"], "glide")
        self.dock_timeout = float(dock_timeout)
        if "timeout" in kwargs.items():
            self.dock_timeout = float(kwargs["timeout"])  # Back compatability
        self.prep_timeout = float(prep_timeout)
        self.variants = None
        self.docking_results = None
        self.temp_dir = TemporaryDirectory()

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

    @staticmethod
    def modify_glide_in(glide_in: str, glide_property: str, glide_value: str):
        """
        Convenience function to insert / overwrite certain .in file properties and values
        :param glide_in: A string of the .in file
        :param property: Property to be changed (e.g. POSE_OUTTYPE)
        :param value: Value to change to (e.g. ligandlib_sd)
        :return: Modified glide_in
        """
        # If property is already present, replace value
        if any([True if glide_property in line else False for line in glide_in]):
            for i in range(len(glide_in)):
                if glide_property in glide_in[i]:
                    glide_in[i] = f"{glide_property}   {glide_value}\n"
                    break
        # Otherwise insert it before newline (separates properties from features)
        elif any([True if line == "\n" else False for line in glide_in]):
            for i in range(len(glide_in)):
                if glide_in[i] == "\n":
                    glide_in.insert(i, f"{glide_property}   {glide_value}\n")
                    break
        # Otherwise just stick it on the end of the file
        else:
            glide_in.append(f"{glide_property}   {glide_value}\n")

        return glide_in

    def run_glide(self):
        """
        Write new input files and submit each to Glide
        """
        glide_commands = []
        for name in self.file_names:
            for variant in self.variants[name]:
                # Set some file paths
                glide_in = self.glide_options.copy()
                # Change glide_in file
                glide_in = self.modify_glide_in(
                    glide_in,
                    "LIGANDFILE",
                    os.path.join(self.directory, f"{name}-{variant}_prepared.sdf"),
                )
                glide_in = self.modify_glide_in(
                    glide_in, "OUTPUTDIR", os.path.join(self.directory)
                )

                # Write new input file (.in)
                with open(
                    os.path.join(self.directory, f"{name}-{variant}.in"), "wt"
                ) as f:
                    [f.write(line) for line in glide_in]

                # Prepare command line command
                command = (
                    f"cd {self.temp_dir.name} ; "
                    + self.glide_env
                    + " -WAIT -NOJOBID -NOLOCAL "
                    + os.path.join(self.directory, f"{name}-{variant}.in")
                )
                glide_commands.append(command)

        # Initialize subprocess
        logger.debug("Glide called")
        p = timedSubprocess(timeout=self.dock_timeout, shell=True).run

        if self.client is not None:
            futures = self.client.map(p, glide_commands)
            results = self.client.gather(futures)
        else:
            results = [p(command) for command in glide_commands]
        logger.debug("Glide finished")
        _ = [logger.debug(err.decode()) for out, err in results if err != "".encode()]
        return self

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

            # If no variants ... next for loop won't run
            if len(self.variants[name]) == 0:
                logger.debug(f"{name}_lib.sdfgz does not exist")
                if best_score[name] is None:  # Only if no other score for prefix
                    docking_result.update(
                        {f"{self.prefix}_" + k: 0.0 for k in self.glide_metrics}
                    )
                    logger.debug("Returning 0.0 as no variants exist")

            # For each variant
            for variant in self.variants[name]:
                out_file = os.path.join(self.directory, f"{name}-{variant}_lib.sdfgz")

                if os.path.exists(out_file):
                    # Try to load it in, and grab the score
                    try:
                        with gzip.open(out_file) as f:
                            glide_out = Chem.ForwardSDMolSupplier(f)

                            for mol in glide_out:  # should just be one
                                dscore = mol.GetPropsAsDict()["r_i_docking_score"]

                                # If molecule doesn't have a score yet append it and the variant
                                if best_score[name] is None:
                                    best_score[name] = dscore
                                    best_variants[i] = f"{name}-{variant}"
                                    docking_result.update(
                                        {
                                            f"{self.prefix}_" + k: v
                                            for k, v in mol.GetPropsAsDict().items()
                                            if k in self.glide_metrics
                                        }
                                    )
                                    # Add charge info
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
                                    logger.debug(
                                        f"Docking score for {name}-{variant}: {dscore}"
                                    )

                                # If docking score is better change it...
                                elif dscore < best_score[name]:
                                    best_score[name] = dscore
                                    best_variants[i] = f"{name}-{variant}"
                                    docking_result.update(
                                        {
                                            f"{self.prefix}_" + k: v
                                            for k, v in mol.GetPropsAsDict().items()
                                            if k in self.glide_metrics
                                        }
                                    )
                                    # Add charge info
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
                                    logger.debug(
                                        f"Found better {name}-{variant}: {dscore}"
                                    )

                                # Otherwise ignore
                                else:
                                    pass

                    # If parsing the molecule threw an error and nothing stored, append 0
                    except Exception:
                        logger.debug(
                            f"Error processing {name}-{variant}_lib.sdfgz file"
                        )
                        if (
                            best_score[name] is None
                        ):  # Only if no other score for prefix
                            best_variants[i] = f"{name}-{variant}"
                            docking_result.update(
                                {f"{self.prefix}_" + k: 0.0 for k in self.glide_metrics}
                            )
                            logger.debug(
                                "Returning 0.0 unless a successful variant is found"
                            )

                # If path doesn't exist and nothing store, append 0
                else:
                    logger.debug(f"{name}-{variant}_lib.sdfgz does not exist")
                    if best_score[name] is None:  # Only if no other score for prefix
                        best_variants[i] = f"{name}-{variant}"
                        docking_result.update(
                            {f"{self.prefix}_" + k: 0.0 for k in self.glide_metrics}
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

        keep_poses = [f"{k}_lib.sdfgz" for k in keep]
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
            os.path.abspath(directory), f"{self.prefix}_GlideDock", step
        )
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
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
        self.run_glide()
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


# class GlideDockFromGlide(GlideDock):
# TODO implement inplace glide docking from previous dock
#    raise NotImplementedError
