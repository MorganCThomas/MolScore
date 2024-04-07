import ast
import atexit
import glob
import logging
import os
from copy import deepcopy
from tempfile import TemporaryDirectory
from typing import Union

from rdkit.Chem import AllChem as Chem

from molscore.scoring_functions._ligand_preparation import ligand_preparation_protocols
from molscore.scoring_functions.descriptors import MolecularDescriptors
from molscore.scoring_functions.utils import DaskUtils, check_openbabel, timedSubprocess

logger = logging.getLogger("gold")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class GOLDDock:
    """
    Score structures using GOLD docking software (by defualt this class uses ChemPLP scoring function)
    1. Jones G, Willett P, Glen RC. Molecular recognition of receptor sites using a genetic algorithm with a description of desolvation. J Mol Biol. 1995 Jan 6;245(1):43-53. doi: 10.1016/s0022-2836(95)80037-9
    2. Jones G, Willett P, Glen RC, Leach AR, Taylor R. Development and validation of a genetic algorithm for flexible docking. J Mol Biol. 1997 Apr 4;267(3):727-48. doi: 10.1006/jmbi.1996.0897
    https://www.ccdc.cam.ac.uk/solutions/csd-discovery/components/gold/
    """

    generic_metrics = ["NetCharge", "PositiveCharge", "NegativeCharge", "best_variant"]
    # This class is ChemPLP by default
    return_metrics = [
        "Score",
        "S(PLP)",
        "S(hbond)",
        "S(cho)",
        "S(metal)",
        "DE(clash)",
        "DE(tors)",
        "time",
    ] + generic_metrics
    docking_metric = "Score"

    @staticmethod
    def check_installation():
        # Check installation
        if "GOLD" not in list(os.environ.keys()):
            raise RuntimeError(
                "GOLD installation not found, please install and add gold_auto to os environment (e.g., export GOLD=<path_to_plants_executable>)"
            )

    # Can additionally add WATER DATA and WRITE OPTIONS
    default_config = {
        "GOLD CONFIGURATION FILE": {},
        "AUTOMATIC SETTINGS": {"autoscale": "1"},
        "POPULATION": {
            "popsiz": "auto",
            "select_pressure": "auto",
            "n_islands": "auto",
            "maxops": "auto",
            "niche_siz": "auto",
        },
        "GENETIC OPERATORS": {
            "pt_crosswt": "auto",
            "allele_mutatewt": "auto",
            "migratewt": "auto",
        },
        "FLOOD FILL": {  # Binding site
            "radius": "10",  # Box size
            "origin": "0 0 0",  # In relation to center
            "do_cavity": "1",
            "floodfill_atom_no": "0",  # Turn off
            "cavity_file": "ligand.mol2",  # Place holder for ref ligand file
            "floodfill_center": "cavity_from_ligand",
        },
        "DATA FILES": {
            "ligand_data_file": "ligand.mol2 10",  # Placeholder for query ligand and n. times docked by GA
            "param_file": "DEFAULT",
            "set_ligand_atom_types": "1",
            "set_protein_atom_types": "0",
            "directory": "output",  # Placeholder for output directory
            "tordist_file": "DEFAULT",
            "make_subdirs": "0",
            "save_lone_pairs": "0",  # Don't save lone pairs, can't read files
            "fit_points_file": "fit_pts.mol2",
            "read_fitpts": "0",
        },
        "FLAGS": {
            "internal_ligand_h_bonds": "1",  # Allow internal H bonds
            "flip_free_corners": "1",  # Allow limited acyclic ring conformational search
            "match_ring_templates": "0",
            "flip_amide_bonds": "0",
            "flip_planar_n": "1 flip_ring_NRR flip_ring_NHR",
            "flip_pyramidal_n": "0",
            "rotate_carboxylic_oh": "flip",
            "use_tordist": "1",
            "postprocess_bonds": "1",
            "rotatable_bond_override_file": "DEFAULT",
            "solvate_all": "1",
        },
        "TERMINATION": {
            "early_termination": "1",  # Enable early stopping based on n_top_solutions and rms_tolerance
            "n_top_solutions": "1",
            "rms_tolerance": "1.5",
        },
        "CONSTRAINTS": {"force_constraints": "0"},
        "COVALENT BONDING": {"covalent": "0"},
        "SAVE OPTIONS": {
            "save_score_in_file": "1",
            "save_protein_torsions": "1",
        },
        "WRITE OPTIONS": {
            "write_options": "NO_LOG_FILES NO_GOLD_LIGAND_MOL2_FILE NO_GOLD_PROTEIN_MOL2_FILE NO_LGFNAME_FILE NO_PLP_MOL2_FILES NO_PID_FILE NO_SEED_LOG_FILE NO_GOLD_ERR_FILE NO_FIT_PTS_FILES NO_ASP_MOL2_FILES NO_GOLD_LOG_FILE"
        },
        "FITNESS FUNCTION SETTINGS": {
            "initial_virtual_pt_match_max": "3",
            "relative_ligand_energy": "0",
            "gold_fitfunc_path": "plp",  # goldscore, chemscore, asp, plp (plp best by CASF-2016)
            "score_param_file": "DEFAULT",
        },
        "PROTEIN DATA": {"protein_datafile": "protein.mol2"},
    }

    def __init__(
        self,
        prefix: str,
        gold_template: Union[None, str, os.PathLike] = None,
        receptor: Union[str, os.PathLike] = None,
        ref_ligand: Union[str, os.PathLike] = None,
        cluster: Union[str, int] = None,
        ligand_preparation: str = "GypsumDL",
        prep_timeout: float = 30.0,
        dock_timeout: float = 120.0,
        **kwargs,
    ):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param gold_template: Template config file, otherwise use default values specified at source
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
        assert (gold_template is not None) or (
            (receptor is not None) and (ref_ligand is not None)
        ), "Must specify gold template config, or both receptor and ref_ligand"

        # Convert any necessary file formats
        self.subprocess = timedSubprocess(shell=True)
        if ref_ligand.endswith(".pdb") or ref_ligand.endswith(".sdf"):
            ext = "." + ref_ligand.split(".")[-1]
            mol2_ligand = ref_ligand.replace(ext, ".mol2")
            self.subprocess.run(
                cmd=f"obabel {ref_ligand} -O {mol2_ligand} --partialcharge none"
            )
            ref_ligand = mol2_ligand

        # Specify class attributes
        self.prefix = prefix.replace(" ", "_")
        self.gold_metrics = self.return_metrics
        self.gold_env = os.environ["GOLD"]
        self.dock_timeout = float(dock_timeout)
        if "timeout" in kwargs.items():
            self.dock_timeout = float(kwargs["timeout"])  # Back compatability
        self.prep_timeout = float(prep_timeout)
        self.gold_template = gold_template
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

        # Set GOLD params
        if self.gold_template:
            self.params = self.read_gold_config(self.gold_template)
        else:
            self.params = self.default_config
        if self.receptor is not None:
            self.params["PROTEIN DATA"]["protein_datafile"] = self.receptor
        if self.ref_ligand is not None:
            self.params["FLOOD FILL"]["cavity_file"] = self.ref_ligand

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
    def read_gold_config(gold_config):
        with open(gold_config, "rt") as f:
            configs = {}
            lines = f.read().splitlines()
            for line in lines:
                if line.startswith(" "):
                    current_configs = line.strip()
                    configs[current_configs] = {}
                else:
                    if line != "":
                        if "=" in line:
                            param = line.split("=")
                        else:
                            param = line.split(" ")
                        configs[current_configs][param[0].strip()] = " ".join(
                            [p.strip() for p in param[1:]]
                        )
        return configs

    @staticmethod
    def write_gold_config(params, output_file):
        join_exceptions = ["ligand_data_file"]
        with open(output_file, "wt") as f:
            for key1 in params.keys():
                f.write(f"  {key1}\n")
                for key2 in params[key1]:
                    if key2 in join_exceptions:
                        f.write(f"{key2} {params[key1][key2]}\n")
                    else:
                        f.write(f"{key2} = {params[key1][key2]}\n")
                f.write("\n")

    @staticmethod
    def read_gold_bestranking(input_file):
        with open(input_file, "rt") as f:
            output = f.read().splitlines()
            last_comment = [
                line for i, line in enumerate(output) if line.startswith("#")
            ][-1]
            keys = [k for k in last_comment.strip("#").split(" ") if k != ""]
            values = [v for v in output[-1].split(" ") if v != ""]
            results = {k: ast.literal_eval(v) for k, v in zip(keys, values)}
            for k in ["File", "name", "File name", "Ligand name"]:
                if k in results.keys():
                    results.pop(k)
        return results

    def reformat_ligands(self, varients, varient_files):
        """Reformat prepared ligands to .mol2"""
        new_varient_files = []
        for vfile in varient_files:
            new_vfile = vfile.replace(".sdf", ".mol2")
            self.subprocess.run(cmd=f"obabel {vfile} -O {new_vfile}")
            new_varient_files.append(new_vfile)
        return varients, new_varient_files

    def run_gold(self):
        """
        Write new config files and submit each to GOLD
        """
        gold_commands = []
        for name in self.file_names:
            for variant in self.variants[name]:
                # Write config file
                params = self.params.copy()
                params["DATA FILES"]["ligand_data_file"] = (
                    os.path.join(self.directory, f"{name}-{variant}_prepared.mol2")
                    + " 10"
                )  # Number of docking poses to return
                params["DATA FILES"]["directory"] = os.path.join(
                    self.directory, f"{name}-{variant}"
                )
                config_file = os.path.join(self.directory, f"{name}-{variant}.conf")
                self.write_gold_config(params=params, output_file=config_file)

                # Prepare GOLD with {gold_env} {config}
                command = f"{self.gold_env} {config_file}"
                gold_commands.append(command)

        # Initialize subprocess
        logger.debug("GOLD called")
        p = timedSubprocess(timeout=self.dock_timeout).run

        # Submit docking subprocesses
        if self.client is not None:
            futures = self.client.map(p, gold_commands)
            _ = self.client.gather(futures)
        else:
            _ = [p(command) for command in gold_commands]
        logger.debug("GOLD finished")
        return self

    def get_docking_scores(self, smiles, return_best_variant=True):
        """
        Extract docking scores from the results
        """
        # Iterate over variants
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
                    self.directory, f"{name}-{variant}", "bestranking.lst"
                )
                if os.path.exists(out_file):
                    try:
                        # Try to load it in, and grab the first line score
                        gold_results = self.read_gold_bestranking(out_file)
                        dscore = gold_results[
                            self.docking_metric
                        ]  # This is a fitness metric so we pose with maximum value

                        # If molecule doesn't have a score yet append it and the variant
                        if (best_score[name] is None) or (dscore > best_score[name]):
                            best_score[name] = dscore
                            best_variants[i] = f"{name}-{variant}"
                            docking_result.update(
                                {
                                    f"{self.prefix}_" + k: v
                                    for k, v in gold_results.items()
                                    if k in self.return_metrics
                                }
                            )
                            logger.debug(f"Best score for {name}-{variant}: {dscore}")
                            # Add charge info
                            try:
                                mol_file = os.path.join(
                                    self.directory,
                                    f"{name}-{variant}",
                                    f"ranked_{name}-{variant}_prepared_m1_1.mol2",
                                )
                                self.subprocess.run(
                                    f'obabel {mol_file} -O {mol_file.replace(".mol2", ".sdf").replace("prepared", "docked")}'
                                )
                                mol_file = mol_file.replace(".mol2", ".sdf").replace(
                                    "prepared", "docked"
                                )
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

        keep_files = [os.path.join(k, f"ranked_{k}_docked_m1_1.sdf") for k in keep] + [
            os.path.join(k, "bestranking.lst") for k in keep
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
            os.path.abspath(directory), f"{self.prefix}_GOLDDock", step
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
        if self.cluster is not None:
            if int(step) % 250 == 0:
                self.client.restart()

        # Run protocol
        self.variants, variant_files = self.ligand_protocol(
            smiles=smiles, directory=self.directory, file_names=file_names
        )
        self.variants, variant_files = self.reformat_ligands(
            self.variants, variant_files
        )
        self.run_gold()
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


class ChemPLPGOLDDock(GOLDDock):
    return_metrics = [
        "Score",
        "S(PLP)",
        "S(hbond)",
        "S(cho)",
        "S(metal)",
        "DE(clash)",
        "DE(tors)",
        "time",
    ] + GOLDDock.generic_metrics
    docking_metric = "Score"
    default_config = deepcopy(GOLDDock.default_config)
    default_config["FITNESS FUNCTION SETTINGS"]["gold_fitfunc_path"] = "plp"


class ASPGOLDDock(GOLDDock):
    return_metrics = [
        "Score",
        "ASP",
        "S(Map)",
        "DE(clash)",
        "DE(int)",
        "time",
    ] + GOLDDock.generic_metrics
    docking_metric = "Score"
    default_config = deepcopy(GOLDDock.default_config)
    default_config["FITNESS FUNCTION SETTINGS"]["gold_fitfunc_path"] = "asp"


class ChemScoreGOLDDock(GOLDDock):
    return_metrics = [
        "Score",
        "DG",
        "S(hbond)",
        "S(metal)",
        "S(lipo)",
        "H(rot)",
        "DE(clash)",
        "DE(int)",
        "time",
    ] + GOLDDock.generic_metrics
    docking_metric = "Score"
    default_config = deepcopy(GOLDDock.default_config)
    default_config["FITNESS FUNCTION SETTINGS"]["gold_fitfunc_path"] = "chemscore"


class GoldScoreGOLDDock(GOLDDock):
    return_metrics = [
        "Fitness",
        "S(hb_ext)",
        "S(vdw_ext)",
        "S(hb_int)",
        "S(int)",
        "time",
    ] + GOLDDock.generic_metrics
    docking_metric = "Fitness"
    default_config = deepcopy(GOLDDock.default_config)
    default_config["FITNESS FUNCTION SETTINGS"]["gold_fitfunc_path"] = "goldscore"
