import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
import shutil
from pathlib import Path
from typing import Optional, Union, List
from contextlib import contextmanager

import numpy as np
import pandas as pd
from rdkit.Chem import AllChem as Chem

import molscore.scaffold_memory as scaffold_memory
import molscore.scoring_functions as scoring_functions
from moleval.metrics.score_metrics import ScoreMetrics
from molscore import resources, utils
from molscore.gui import monitor_path

logger = logging.getLogger("molscore")
logger.setLevel(logging.WARNING)

PRESETS = {
        "GuacaMol": resources.files("molscore.configs.GuacaMol"),
        "GuacaMol_Scaffold": resources.files("molscore.configs.GuacaMol_Scaffold"),
        "MolOpt": resources.files("molscore.configs.MolOpt"),
        "MolExp": resources.files("molscore.configs.MolExp"),
        "MolExp_baseline": resources.files("molscore.configs.MolExp_baseline"),
        "MolExpL": resources.files("molscore.configs.MolExpL"),
        "MolExpL_baseline": resources.files("molscore.configs.MolExpL_baseline"),
        "MolOpt-CF": resources.files("molscore.configs.MolOpt-CF"),
        "MolOpt-DF": resources.files("molscore.configs.MolOpt-DF"),
        "5HT2A_PhysChem": resources.files("molscore.configs.5HT2A.PhysChem"),
        "5HT2A_Selectivity": resources.files(
            "molscore.configs.5HT2A.PIDGIN_Selectivity"
        ),
        "5HT2A_Docking": resources.files(
            "molscore.configs.5HT2A.Docking_Selectivity_rDock"
        ),
        "LibINVENT_Exp1": resources.files("molscore.configs.LibINVENT"),
        "LinkINVENT_Exp3": resources.files("molscore.configs.LinkINVENT"),
        "3D_Benchmark": resources.files("molscore.configs.3D_Benchmark"),  
    }


class MolScore:
    """
    Central manager class that, when called, takes in a list of SMILES and returns respective scores.
    """
    presets = PRESETS

    preset_tasks = {
        k:[p.stem for p in v.glob("*.json")] 
        for k, v in presets.items()
        }
    
    @staticmethod
    def load_config(task_config: os.PathLike, diversity_filter: str = None) -> dict:
        assert os.path.exists(
            task_config
        ), f"Configuration file {task_config} doesn't exist"
        with open(task_config, "r") as f:
            configs = f.read().replace("\r", "").replace("\n", "").replace("\t", "")
        configs = json.loads(configs)
        if diversity_filter is not None:
            df_presets = resources.files("molscore.configs._diversity_filters").glob("*.json")
            df_config = None
            for p in df_presets:
                name = p.stem
                if diversity_filter == name:
                    with open(p, "r") as f:
                        df_config = json.load(f)
                    configs["diversity_filter"] = df_config
                    break
            if df_config is None:
                raise ValueError(
                    f"Could not find diversity filter {diversity_filter} in {df_presets}"
                )
        return configs
    
    @staticmethod
    def parse_path(path: Union[os.PathLike, List[str]] = None) -> os.PathLike:
        "Convert path or resource-like path to absolute path"
        if path is None:
            return None
        if isinstance(path, list):
            # It's a resource type
            assert len(path) == 2, f"Path {path} is not a valid resource-like path containing two items"
            return str(resources.files(path[0]).joinpath(path[1]))
        else:
            # It's path like
            assert os.path.exists(path), f"Path {path} doesn't exist"
            return os.path.abspath(path)

    def __init__(
        self,
        model_name: str,
        task_config: Union[str, os.PathLike],
        output_dir: str = None,
        add_run_dir: bool = True,
        run_name: str = None,
        budget: int = None,
        oracle_budget: int = None,
        termination_threshold: int = None,
        termination_patience: int = None,
        termination_exit: bool = False,
        score_invalids: bool = False,
        replay_size: int = None,
        replay_purge: bool = True,
        diversity_filter: str = None,
        **kwargs,
    ):
        """
        :param model_name: Name of generative model, used for file naming and documentation
        :param task_config: Path to task config file, or a preset name e.g., GuacaMol:Albuterol_similarity
        :param output_dir: Overwrites the output directory specified in the task config file
        :param add_run_dir: Adds a run directory within the output directory
        :param run_name: Override the run name with a custom name, otherwise taken from 'task' in the config
        :param budget: Maximum number of molecules for MolScore before molscore.finished is True
        :param oracle_budget: Maximum number of unique molecules actually passed to oracles before molscore.finished is True
        :param termination_threshold: Threshold for early stopping based on the score
        :param termination_patience: Number of steps with no improvement, or that a termination_threshold has been reached for
        :param termination_exit: Exit on termination of objective
        :param score_invalids: Whether to force scoring of invalid molecules
        :param replay_size: Maximum size of the replay buffer
        :param replay_purge: Whether to purge the replay buffer, i.e., only allow molecules that pass the diversity filter
        :param diversity_filter: Name a diversity filter to add/overide the one in the config
        """
        # Load in configuration file (json)
        if task_config.endswith(".json"):
            self.cfg = self.load_config(task_config, diversity_filter=diversity_filter)
        else:
            assert ":" in task_config, "Preset task must be in format 'category:task'"
            cat, task = task_config.split(":", maxsplit=1)
            assert cat in self.presets.keys(), f"Preset category {cat} not found"
            task_config = self.presets[cat] / f"{task}.json"
            assert task_config.exists(), f"Preset task {task} not found in {cat}"
            self.cfg = self.load_config(task_config, diversity_filter=diversity_filter)

        # Here are attributes used
        self.model_name = model_name.replace(" ", "_")
        self.name = self.cfg["task"].replace(" ", "_")
        self.step = 0
        self.current_idx = 0
        self.budget = budget
        self.oracle_budget = oracle_budget
        self.termination_threshold = termination_threshold
        self.termination_patience = termination_patience
        reset_termination_criteria = not any(
            [budget, oracle_budget, termination_threshold, termination_patience]
        )
        self.termination_counter = 0
        self.termination_exit = termination_exit
        self.score_invalids = score_invalids
        self.replay_size = replay_size
        self.replay_purge = replay_purge
        self.replay_buffer = utils.ReplayBuffer(size=replay_size, purge=replay_purge)
        self.replay_buffer = utils.ReplayBuffer(size=replay_size, purge=replay_purge)
        self.finished = False
        self.init_time = time.time()
        self.results_df = None # TODO make empty df ... 
        self.batch_df = None
        self.exists_map = {}
        self.main_df = None
        self.monitor_app = None
        self.diversity_filter = None
        self.call2score_warning = True
        self.metrics = None
        self.starting_population = self.parse_path(self.cfg.get("starting_population", None))
        self.logged_parameters = {}  # Extra parameters to write out in scores.csv for comparative purposes
        self.temp_parameters = {} # Temp parameters to write out each iteration in scores.csv for comparative purposes

        # Setup save directory
        if not run_name:
            run_name = self.cfg["task"].replace(" ", "_")
        self.run_name = "_".join(
            [
                self.model_name,
                run_name,
                time.strftime("%Y_%m_%d", time.localtime()),
            ]
        )
        if output_dir is not None:
            self.save_dir = os.path.abspath(output_dir)
        else:
            self.save_dir = os.path.abspath(self.cfg["output_dir"])
        if add_run_dir:
            self.save_dir = os.path.join(self.save_dir, self.run_name)
        
        # Check to see if we're loading from previous results
        if self.cfg["load_from_previous"]:
            assert os.path.exists(
                self.cfg["previous_dir"]
            ), "Previous directory does not exist"
            self.save_dir = self.cfg["previous_dir"]
            
        # Create save directory
        else:
            if os.path.exists(self.save_dir) and add_run_dir:
                print(
                    "Found existing directory, appending current time to distinguish"
                )  # Not set up logging yet
                self.save_dir = self.save_dir + time.strftime(
                    "_%H_%M_%S", time.localtime()
                )
            os.makedirs(self.save_dir, exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, "iterations"))

        # Setup log file
        self.fh = logging.FileHandler(os.path.join(self.save_dir, "log.txt"))
        self.fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        self.fh.setFormatter(formatter)
        logger.addHandler(self.fh)
        ch = logging.StreamHandler()
        try:
            if self.cfg["logging"]:
                ch.setLevel(logging.INFO)
            else:
                ch.setLevel(logging.WARNING)
            logger.addHandler(ch)
        except KeyError:
            ch.setLevel(logging.WARNING)
            logger.addHandler(ch)
            logger.info(
                "Verbose logging unspecified, defaulting to non-verbose... so how are seeing this?"
            )
            pass

        # Setup monitor app
        try:
            if self.cfg["monitor_app"]:
                self.monitor_app = True
                self.monitor_app_path = monitor_path
        except KeyError:
            logger.info("Run monitor option unspecified, defaulting to False")
            self.monitor_app = False
            # For backwards compatibility, default too false
            pass

        # Load modifiers/tranformations
        self.modifier_functions = utils.all_score_modifiers

        # Set scoring function / transformation / aggregation / diversity filter
        self._set_objective(
            reset_diversity_filter=True,
            reset_termination_criteria=reset_termination_criteria,
        )

        # Load from previous
        if self.cfg["load_from_previous"]:
            logger.info("Loading scores.csv from previous run")
            self.main_df = pd.read_csv(
                os.path.join(self.save_dir, "scores.csv"),
                index_col=0,
            )
            logger.debug(self.main_df.head())
            # Update step and idx
            self.step = max(self.main_df["step"])
            self.current_idx = max(self.main_df.index[-1] + 1)
            # Update time
            self.init_time = time.time() - self.main_df["absolute_time"].iloc[-1]
            # Update max min
            self.update_maxmin(df=self.main_df)
            # Load in diversity filter
            if os.path.exists(os.path.join(self.save_dir, "scaffold_memory.csv")):
                assert isinstance(
                    self.diversity_filter, scaffold_memory.ScaffoldMemory
                ), (
                    "Found scaffold_memory.csv but diversity filter seems to not be ScaffoldMemory type"
                    "are you running the same diversity filter as previously?"
                )
                logger.info("Loading scaffold_memory.csv from previous run")
                previous_scaffold_memory = pd.read_csv(
                    os.path.join(self.save_dir, "scaffold_memory.csv")
                )
                self.diversity_filter._update_memory(
                    smiles=previous_scaffold_memory["SMILES"].tolist(),
                    scaffolds=previous_scaffold_memory["Scaffold"].tolist(),
                    scores=previous_scaffold_memory.loc[
                        :,
                        ~previous_scaffold_memory.columns.isin(
                            ["Cluster", "Scaffold", "SMILES"]
                        ),
                    ].to_dict("records"),
                )
            # Load in replay buffer
            if os.path.exists(os.path.join(self.save_dir, "replay_buffer.csv")):
                logger.info("Loading replay_buffer.csv from previous run")
                self.replay_buffer.load(
                    os.path.join(self.save_dir, "replay_buffer.csv")
                )
        logger.info("MolScore initiated")
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """ Context teardown """
        self.write_scores()
        self.kill_monitor()
        if exc_type:
            print(f"Handled exception: {exc_value}")
        return False

    def _set_objective(
        self,
        task_config: str = None,
        reset_diversity_filter: bool = True,
        reset_termination_criteria: bool = False,
        reset_replay_buffer: bool = False,
    ):
        """
        Set or reset the overall configuration for scoring, but not logging.
        """
        if task_config:
            # Load in configuration file (json)
            self.cfg = self.load_config(task_config)

        # Write config
        with open(
            os.path.join(self.save_dir, f"{self.cfg['task']}_config.json"), "w"
        ) as config_f:
            json.dump(self.cfg, config_f, indent=2)

        # Set / reset scoring functions
        self.scoring_functions = []
        for fconfig in self.cfg["scoring_functions"]:
            if fconfig["run"]:
                for fclass in scoring_functions.all_scoring_functions:
                    if fclass.__name__ == fconfig["name"]:
                        self.scoring_functions.append(fclass(**fconfig["parameters"]))
                if all(
                    [
                        fclass.__name__ != fconfig["name"]
                        for fclass in scoring_functions.all_scoring_functions
                    ]
                ):
                    logger.warning(
                        f'Not found associated scoring function for {fconfig["name"]}'
                    )
            else:
                pass
        if len(self.scoring_functions) == 0:
            logger.warning("No scoring functions assigned")
            
        # Set / reset
        self.starting_population = self.parse_path(self.cfg.get("starting_population", None))

        # Setup aggregation/mpo methods
        for func in utils.all_score_methods:
            if self.cfg["scoring"]["method"] == func.__name__:
                self.mpo_method = func
        assert any(
            [
                self.cfg["scoring"]["method"] == func.__name__
                for func in utils.all_score_methods
            ]
        ), "No aggregation methods not found"

        # Setup Diversity filters
        if reset_diversity_filter:
            try:
                if self.cfg["diversity_filter"]["run"]:
                    self.diversity_filter = self.cfg["diversity_filter"]["name"]
                    if self.diversity_filter not in [
                        "Unique",
                        "Occurrence",
                    ]:  # Then it's a memory-assisted one
                        for filt in scaffold_memory.all_scaffold_filters:
                            if self.cfg["diversity_filter"]["name"] == filt.__name__:
                                self.diversity_filter = filt(
                                    **self.cfg["diversity_filter"]["parameters"]
                                )
                        if all(
                            [
                                filt.__name__ != self.cfg["diversity_filter"]["name"]
                                for filt in scaffold_memory.all_scaffold_filters
                            ]
                        ):
                            logger.warning(
                                f'Not found associated diversity filter for {self.cfg["diversity_filter"]["name"]}'
                            )
                    self.log_parameters(
                        {
                            "diversity_filter": self.cfg["diversity_filter"]["name"],
                            "diversity_filter_params": self.cfg["diversity_filter"][
                                "parameters"
                            ],
                        }
                    )
            except KeyError:
                # Backward compatibility if diversity filter not found, default to not run one...
                self.diversity_filter = None
                pass

        # Set / reset budget termination criteria
        if reset_termination_criteria:
            if self.cfg.get("budget", None):
                self.budget = self.cfg.get("budget", None)
            if self.cfg.get("oracle_budget", None):
                self.oracle_budget = self.cfg.get("oracle_budget", None)
            if self.cfg.get("termination_threshold", None):
                self.termination_threshold = self.cfg.get("termination_threshold", None)
            if self.cfg.get("termination_patience", None):
                self.termination_patience = self.cfg.get("termination_patience", None)
            if self.cfg.get("termination_exit", None):
                self.termination_exit = self.cfg.get("termination_exit", False)
            self.termination_counter = 0

        # Reset replay buffer
        if reset_replay_buffer:
            self.replay_buffer.reset()

        # Add warning in case of possible neverending optimization
        if self.termination_threshold and not self.budget:
            logger.warning(
                "Termination threshold set but no budget specified, this may result in never-ending optimization if threshold is not reached."
            )

    def parse(
        self,
        step: int,
        mol_ids: Optional[list] = None,
        smiles: Optional[list] = None,
        canonicalise_smiles: bool = True,
        check_uniqueness: bool = True,
        **molecular_inputs,
    ):
        """
        Create batch_df object from initial list of molecular inputs and calculate validity and
        intra-batch uniqueness if possible.

        :param step: Current generative model step
        :param mol_ids: List of molecular identifiers, otherwise the numeric index will be used.
        :param smiles: List of smiles from a generative model
        :param canonicalise_smiles: Whether to canonicalise smiles using RDKit
        :param check_uniqueness: Whether to check for uniqueness 
        """
        # Initialize df for batch
        _batch_size = len(smiles) if smiles else len(list(molecular_inputs.values())[0])
        _running_idx = list(range(self.current_idx, self.current_idx + _batch_size))
        _batch_idx = list(range(_batch_size))
        self.batch_df = pd.DataFrame(index=_running_idx)

        # Add batch constants
        self.batch_df["model"] = self.model_name.replace(" ", "_")
        self.batch_df["task"] = self.cfg["task"].replace(" ", "_")
        self.batch_df["step"] = step
        self.batch_df["absolute_time"] = time.time() - self.init_time

        # Add indexes
        self.batch_df["batch_idx"] = _batch_idx
        self.batch_df["mol_id"] = mol_ids if mol_ids else _running_idx
        
        # NOTE/TODO: How should we check or bypass validity for 3D (inorganic) molecules?
        # For now, we assume all 3D molecules are valid
        self.batch_df["valid"] = True
        self.batch_df["valid_score"] = 1

        # Parse SMILES if present
        if smiles:
            parsed_smiles = []
            valid = []
            for smi in smiles:
                if not smi: # If None or empty string
                    parsed_smiles.append(None)
                    valid.append(False)
                    continue
                    
                if canonicalise_smiles:
                    try:
                        can_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
                        parsed_smiles.append(can_smi)
                        valid.append(True)
                    except TypeError:
                        try:
                            # Try to catch invalid molecules and sanitize
                            mol = Chem.MolFromSmiles(smi)
                            Chem.SanitizeMol(mol)
                            can_smi = Chem.MolToSmiles(mol)
                            parsed_smiles.append(can_smi)
                            valid.append(True)
                        except Exception:
                            parsed_smiles.append(smi)
                            valid.append(False)

                else:
                    parsed_smiles.append(smi)
                    valid.append(True) # Assume valid

            self.batch_df["smiles"] = parsed_smiles
            # Maybe set mol_id as canonical SMILES
            if smiles and canonicalise_smiles and not mol_ids:
                self.batch_df["mol_id"] = self.batch_df["smiles"].copy()
            # Update for SMILES validity
            self.batch_df["valid"] = valid
            self.batch_df["valid_score"] = [1 if v else 0 for v in valid]
            logger.debug(f"    Invalid molecules: {(~self.batch_df.valid).sum()}")
            
        # Check for duplicates and count occurrences
        if check_uniqueness:
            self.batch_df["unique"] = True
            self.batch_df["occurrences"] = 0

            for idx, mol_id in zip(self.batch_df.index, self.batch_df["mol_id"]):
                if mol_id in self.exists_map:
                    self.batch_df.at[idx, "unique"] = False
                    self.batch_df.at[idx, "occurrences"] = self.exists_map[mol_id]["count"]
                    # Update exists_map
                    self.exists_map[mol_id]["count"] += 1
                else:
                    self.exists_map[mol_id] = {"first_idx": idx, "count": 1}


    def run_scoring_functions(
        self, batch_index: list, file_names: list,  mol_ids: Optional[list] = None, **molecular_inputs,):
        """
        Iterate over respective scoring/fitness functions to score molecular inputs and populate self.results_df
         (file_names are necessary for logging).

        :param batch_index: Batch index of the molecular inputs being scored
        :param file_names: A corresponding list of file prefixes for tracking - format={step}_{batch_idx}
        :param mol_ids: List of molecular identifiers.
        :param molecular_inputs: Molecular_inputs to pass to scoring functions
        :return: None
        """
        self.results_df = pd.DataFrame(batch_index, columns=["batch_idx"])
        if mol_ids:
            self.results_df["mol_id"] = mol_ids
        for function in self.scoring_functions:
            results = function(
                directory=self.save_dir, file_names=file_names, **molecular_inputs
            )
            results_df = pd.DataFrame(results).assign(batch_idx=batch_index)
            if mol_ids:
                results_df["mol_id"] = mol_ids
            # Merge
            self.results_df = self.results_df.merge(results_df, how="outer", sort=False)

        # Drop any duplicates in results
        if mol_ids:
            self.results_df = self.results_df.drop_duplicates(subset="mol_id")

    def first_update(self):
        """
        Merge results_df with batch_df. Only used for the first step/batch.
        """
        logger.debug("    Merging results to batch df")
        # Store original index
        original_index = self.batch_df.index
        # Drop overlapping columns before merge (except mol_id)
        columns_to_drop = [
            col for col in self.results_df.columns
            if col in self.batch_df.columns and col != "mol_id"
        ]
        self.results_df.drop(columns=columns_to_drop, axis=1, inplace=True)
        # Merge into batch_df
        self.batch_df = self.batch_df.merge(self.results_df, on='mol_id', how="left", sort=False).infer_objects()
        # Reassign original index
        self.batch_df.index = original_index
        self.batch_df.fillna(0.0, inplace=True)

    def concurrent_update(self):
        """
        Merge results_df with batch_df and look up duplicated entries to avoid re-calculating.
        :param mol_ids: If provided will check by mol_id.
        """
        # Get duplicated entries
        duplicate_mol_ids = set(self.batch_df.loc[~self.batch_df["unique"], "mol_id"].unique())
        results_mol_ids = set(self.results_df["mol_id"].unique())
        duplicate_mol_ids = duplicate_mol_ids - results_mol_ids
        # Get results columns
        results_columns = self.results_df.columns
        
        if len(duplicate_mol_ids) > 0:
            # Get main idxs of duplicates mol_ids
            dup_idxs = [self.exists_map[mol_id]["first_idx"] for mol_id in duplicate_mol_ids]
            # Drop any idxs equal or above self.current_idx not in main_df yet, these will be populated in merge
            dup_idxs = [idx for idx in dup_idxs if idx < self.current_idx]  
            
            # Get main data and merge with results
            exists_df = self.main_df.loc[dup_idxs, results_columns]
            self.results_df = pd.concat(
                [self.results_df, exists_df],
                axis=0,
                ignore_index=True,
                sort=False,
            )
        
        # Drop overlapping columns before merge (except mol_id)
        columns_to_drop = [
            col for col in results_columns
            if col in self.batch_df.columns and col != "mol_id"
        ]
        self.results_df.drop(columns=columns_to_drop, axis=1, inplace=True)
        
        # Merge into batch_df
        logger.debug("    Merging results to batch df")
        original_index = self.batch_df.index
        self.batch_df = self.batch_df.merge(
            self.results_df, on="mol_id", how="left", sort=False
        ).infer_objects()
        self.batch_df.index = original_index
        self.batch_df.fillna(0.0, inplace=True)

    def update_maxmin(self, df):
        """
        This function keeps track of maximum and minimum values seen per metric for normalization purposes.

        :return:
        """
        for metric in self.cfg["scoring"]["metrics"]:
            if metric["name"] in df.columns:
                df_max = df.loc[:, metric["name"]].max()
                df_min = df.loc[:, metric["name"]].min()

                if "max" not in metric["parameters"].keys():
                    metric["parameters"].update({"max": df_max})
                    logger.debug(f"    Updated {metric['name']} max to {df_max}")
                elif df_max > metric["parameters"]["max"]:
                    metric["parameters"].update({"max": df_max})
                    logger.debug(f"    Updated {metric['name']} max to {df_max}")
                else:
                    pass

                if "min" not in metric["parameters"].keys():
                    metric["parameters"].update({"min": df_min})
                    logger.debug(f"    Updated {metric['name']} min to {df_min}")
                elif df_min < metric["parameters"]["min"]:
                    metric["parameters"].update({"min": df_min})
                    logger.debug(f"    Updated {metric['name']} min to {df_min}")
                else:
                    pass

    def compute_score(self, df):
        """
        Compute the final score i.e. combination of which metrics according to which method.

        :param df: DataFrame containing the scores for each metric
        """
        mpo_columns = {"names": [], "weights": []}
        filter_columns = {"names": []}
        transformed_columns = {}

        # Iterate through scoring parameters/metrics, transform and aggregate
        for metric in self.cfg["scoring"]["metrics"]:
            mod_name = f"{metric['modifier']}_{metric['name']}"

            # Store parameter as filter (multiply final fitness function)
            if metric.get("filter", False):
                filter_columns["names"].append(mod_name)
            # Store parameter for aggregation into fitness function
            else:
                mpo_columns["names"].append(mod_name)
                mpo_columns["weights"].append(metric["weight"])

            for mod in self.modifier_functions:
                if metric["modifier"] == mod.__name__:
                    modifier = mod

            # Check the modifier function exists
            assert any(
                [metric["modifier"] == mod.__name__ for mod in self.modifier_functions]
            ), f"Score modifier {metric['modifier']} not found"

            # Check the metric can be found in the dataframe
            try:
                assert (
                    metric["name"] in df.columns
                ), f"Specified metric {metric['name']} not found in dataframe"
            except AssertionError as e:
                self._write_temp_state(step=self.step)
                raise e

            # Apply transform to parameter
            transformed_columns[mod_name] = (
                df.loc[:, metric["name"]]
                .apply(lambda x: modifier(x, **metric["parameters"])) 
                .rename(mod_name)
            )

        # Merge transformed parameters into main dataframe
        df = pd.concat([df] + list(transformed_columns.values()), axis=1)

        # Double check we have no NaN or 0 values (necessary for geometric mean) for mpo columns
        df.loc[:, mpo_columns["names"]].fillna(1e-6, inplace=True)
        df[mpo_columns["names"]] = df[mpo_columns["names"]].apply(
            lambda x: [1e-6 if y < 1e-6 else y for y in x]
        )

        # Aggregate into final fitness score
        if mpo_columns["names"]:
            df[self.cfg["scoring"]["method"]] = df.loc[:, mpo_columns["names"]].apply(
                lambda x: self.mpo_method(
                    x=x,
                    w=np.asarray(mpo_columns["weights"]),
                    X=df.loc[:, mpo_columns["names"]].to_numpy(),
                ),
                axis=1,
                raw=True,
            )
        else:
            df[self.cfg["scoring"]["method"]] = 1.0
            logger.warning(
                "No score columns provided for aggregation, returning a full reward of 1.0"
            )

        # Combine filter parameters (anticipated to be 1 or 0)
        df["filter"] = df.loc[:, filter_columns["names"]].apply(
            lambda x: np.prod(x), axis=1, raw=True
        )
        # Multiply fitness score by any filter parameters
        df[self.cfg["scoring"]["method"]] = (
            df[self.cfg["scoring"]["method"]] * df["filter"]
        )
        # Copy to an obvious Score column
        df['Score'] = df[self.cfg["scoring"]["method"]].copy()

        return df

    def run_diversity_filter(self, df):
        if self.diversity_filter == "Unique":
            assert (
                "unique" in df.columns
            ), "Unique column not found in dataframe, cannot apply unique diversity filter"
            df[f"filtered_{self.cfg['scoring']['method']}"] = [
                s if u else 0.0
                for u, s in zip(df["unique"], df[self.cfg["scoring"]["method"]])
            ]
            df["passes_diversity_filter"] = [
                True if float(a) == float(b) else False
                for b, a in zip(
                    df[self.cfg["scoring"]["method"]],
                    df[f"filtered_{self.cfg['scoring']['method']}"],
                )
            ]

        elif self.diversity_filter == "Occurrence":
            assert (
                "occurrences" in df.columns
            ), "occurrences column not found in dataframe, cannot apply occurrences diversity filter"
            df[f"filtered_{self.cfg['scoring']['method']}"] = [
                s
                * utils.lin_thresh(
                    x=o,
                    objective="minimize",
                    upper=0,
                    lower=self.cfg["diversity_filter"]["parameters"]["tolerance"],
                    buffer=self.cfg["diversity_filter"]["parameters"]["buffer"],
                )
                for o, s in zip(df["occurrences"], df[self.cfg["scoring"]["method"]])
            ]
            df["passes_diversity_filter"] = [
                True if float(a) == float(b) else False
                for b, a in zip(
                    df[self.cfg["scoring"]["method"]],
                    df[f"filtered_{self.cfg['scoring']['method']}"],
                )
            ]

        else:  # Memory-assisted
            assert (
                "smiles" in df.columns
            ), "smiles column not found in dataframe, cannot apply memory-assisted diversity filter"
            scores_dict = {
                "total_score": np.asarray(
                    df[self.cfg["scoring"]["method"]].tolist(), dtype=np.float64
                ),
                "step": [self.step] * len(df),
            }
            filtered_scores = self.diversity_filter.score(
                smiles=df["smiles"].tolist(), scores_dict=scores_dict
            )
            df["passes_diversity_filter"] = [
                True if float(a) == float(b) else False
                for b, a in zip(df[self.cfg["scoring"]["method"]], filtered_scores)
            ]
            df[f"filtered_{self.cfg['scoring']['method']}"] = filtered_scores
        
        # Copy to obvious score column
        df['Score (reshaped)'] = df[f"filtered_{self.cfg['scoring']['method']}"].copy()
        df.fillna(1e-6)
        
        return df

    def log_parameters(self, parameters: dict):
        self.logged_parameters.update(parameters)

    def write_scores(self):
        """
        Write final dataframe to file.
        """
        if self.main_df is not None:
            # Save scores.csv
            if len(self.logged_parameters) > 0:
                temp = self.main_df.copy()
                for p, v in self.logged_parameters.items():
                    try:
                        temp[p] = v
                    except ValueError:
                        # temp[p] = [v]*len(temp)
                        pass
                temp.to_csv(os.path.join(self.save_dir, "scores.csv"))  # save main csv
            else:
                self.main_df.to_csv(
                    os.path.join(self.save_dir, "scores.csv")
                )  # save main csv
            
            # Delete iterations dir to help cleanup
            if os.path.exists(os.path.join(self.save_dir, "iterations")):
                shutil.rmtree(os.path.join(self.save_dir, "iterations"))

        if (self.diversity_filter is not None) and (
            isinstance(self.diversity_filter, scaffold_memory.ScaffoldMemory)
        ):
            self.diversity_filter.savetocsv(
                os.path.join(self.save_dir, "scaffold_memory.csv")
            )
        if len(self.replay_buffer) > 0:
            self.replay_buffer.save(os.path.join(self.save_dir, "replay_buffer.csv"))

        self.fh.close()

    def _write_temp_state(self, step):
        try:
            self.main_df.to_csv(os.path.join(self.save_dir, f"scores_{step}.csv"))
            if (self.diversity_filter is not None) and (
                not isinstance(self.diversity_filter, str)
            ):
                self.diversity_filter.savetocsv(
                    os.path.join(self.save_dir, f"scaffold_memory_{step}.csv")
                )
            if len(self.replay_buffer) > 0:
                self.replay_buffer.save(
                    os.path.join(self.save_dir, f"replay_buffer_{step}.csv")
                )
            self.batch_df.to_csv(os.path.join(self.save_dir, f"batch_df_{step}.csv"))
            self.results_df.to_csv(
                os.path.join(self.save_dir, f"results_df_{step}.csv")
            )
        except AttributeError:
            pass  # Some may be NoneType like results
        return

    def _write_attributes(self):
        dir = os.path.join(self.save_dir, "molscore_attributes")
        os.makedirs(dir, exist_ok=True)
        prims = {}
        # If list, dict or csv write to appropriate format
        for k, v in self.__dict__.items():
            if k == "fh":
                continue
            # Convert class to string
            elif k == "scoring_functions":
                with open(os.path.join(dir, k), "wt") as f:
                    nv = [str(i.__class__) for i in v]
                    json.dump(nv, f, indent=2)
            # Convert functions to string
            elif k == "diversity_filter":
                prims.update({k: v.__class__})
            elif k == "modifier_functions":
                continue
            elif k == "mpo_method":
                prims.update({k: str(v.__name__)})
            # Else do it on type
            elif isinstance(v, (list, dict)):
                with open(os.path.join(dir, k), "wt") as f:
                    json.dump(v, f, indent=2)
            elif isinstance(v, pd.core.frame.DataFrame):
                with open(os.path.join(dir, k), "wt") as f:
                    v.to_csv(f)
            else:
                prims.update({k: v})

        # Else write everything else to text
        with open(os.path.join(dir, "single_attributes.txt"), "wt") as f:
            _ = [f.write(f"{k}: {v}\n") for k, v in prims.items()]
        return

    def run_monitor(self):
        """
        Run streamlit monitor.
        """
        # Start dash_utils monitor (Killed in write scores method)
        cmd = ["streamlit", "run", self.monitor_app_path, self.save_dir]
        self.monitor_app = subprocess.Popen(cmd, preexec_fn=os.setsid)

    def kill_monitor(self):
        """
        Kill streamlit monitor.
        """
        if (self.monitor_app is None) or (not self.monitor_app):
            logger.info("No monitor to kill")
        else:
            try:
                os.killpg(os.getpgid(self.monitor_app.pid), signal.SIGTERM)
                _, _ = self.monitor_app.communicate()
            except AttributeError as e:
                logger.error(f"Monitor may not have opened/closed properly: {e}")
            self.monitor_app = None

    def evaluate_finished(self):
        """
        Check if the current task is finished based on budget or termination criteria
        """
        task_df = self.main_df.loc[self.main_df.task == self.cfg["task"]]

        # Based on molecule budget
        if self.budget and (len(task_df) >= self.budget):
            self.finished = True
            return
        
        # Based on oracle budget (i.e., valid and unique)
        if self.oracle_budget and (len(task_df.loc[task_df.valid & task_df.unique]) >= self.oracle_budget):
            self.finished = True
            return

        # Based on patience
        if self.termination_patience and not self.termination_threshold:
            if (
                self.batch_df[self.cfg["scoring"]["method"]].mean()
                < task_df[self.cfg["scoring"]["method"]]
                .rolling(window=500)
                .mean()
                .iloc[-1]
            ):
                self.termination_counter += 1

        # Based on a threshold
        if self.termination_threshold:
            if (
                self.batch_df[self.cfg["scoring"]["method"]].mean()
                >= self.termination_threshold
            ):
                if self.termination_patience:
                    self.termination_counter += 1
                else:
                    self.finished = True
                    return

        # Check our patience value
        if self.termination_patience:
            if self.termination_counter >= self.termination_patience:
                self.finished = True
                return

        if self.termination_exit and self.finished:
            sys.exit(1)

    def _score_only(
        self,
        step: int = None,
        flt: bool = False,
        **molecular_inputs,
        ):
        """
        This hidden method will only score inputs without caching the data.
        """
        
        # Set some values
        batch_start = time.time()
        if step is not None:
            self.step = step
        else:
            self.step += 1
        _batch_size = [len(v) for v in molecular_inputs.values()][0]
        logger.info(f"   Scoring: {_batch_size} molecular inputs")
        
        # Pass inputs to scoring function
        _process_batch_idxs = list(range(_batch_size))
        file_names = [f"{self.step}_{i}" for i in _process_batch_idxs]
        self.run_scoring_functions(
            batch_index=_process_batch_idxs,
            file_names=file_names,
            **molecular_inputs,
        )
        logger.info(
            f"    Score returned for {len(self.results_df)} mols in {time.time() - batch_start:.02f}s"
        )
        
        # Compute final fitness score
        self.update_maxmin(df=self.results_df)
        self.results_df = self.compute_score(self.results_df)
        
        # Apply diversity filter if specified
        if self.diversity_filter is not None:
            self.results_df = self.run_diversity_filter(self.results_df)
            score_col = "Score (reshaped)"
        else:
            score_col = "Score"
            
        # Fetch scores
        scores = self.results_df.loc[:, score_col].tolist()
        if not flt:
            scores = np.array(scores, dtype=np.float32)
        logger.info(
            f"    Score returned for {len(self.results_df)} molecular inputs in {time.time() - batch_start:.02f}s"
        )

        # Clean up class
        self.batch_df = None
        self.results_df = None
        return scores

    def score(
        self,
        smiles: Optional[list] = None,
        mol_ids: Optional[list] = None,
        step: int = None,
        flt: bool = False,
        canonicalise_smiles: bool = True,
        recalculate: bool = False,
        check_uniqueness: bool = True,
        score_only: bool = False,
        **molecular_inputs,
    ):
        """
        Calling this method will result in the primary function of scoring molecular inputs and logging data in
         an automated fashion, and returning output values.

        :param smiles: A list of smiles to score.
        :param mol_ids: A list of molecular identifiers, if not provided the numeric index will be used.
        :param step: Step of generative model for logging, and indexing. This could equally be iterations/epochs etc.
        :param flt: Whether to return a list of floats (default False i.e. return np.array of type np.float32)
        :param canonicalise_smiles: Whether to canonicalise smiles during parsing
        :param recalculate: Whether to pass duplicated mol_ids to the scoring functions again,
          may be desirable if the scoring function is somewhat stochastic.
          (default False i.e. re-use existing scores for duplicated molecules to save)
        :param check_uniqueness: Whether to check for uniqueness of the input molecules
        :param score_only: Whether to cache molecular data or simply score inputs and return scores
        :param molecular_inputs: Score accepts arbitrary keyword arguments of molecular representations that will be passed to scoring functions
        :return: Scores (either float list or np.array)

        Examples:
            >>> from molscore import MolScore, MockGenerator

            # Scoring smiles
            >>> MS = MolScore(model_name='test', task_config='GuacaMol:Albuterol_similarity')
            >>> MG = MockGenerator()
            >>> smiles = MG.sample(10)
            >>> scores = MS.score(smiles=smiles)

            # Passing other representations e.g., RDKit Mol objects
            >>> mols = [Chem.MolFromSmiles(smi) for smi in smiles]
            >>> scores = MS.score(rdkit_mols=mols) # Scoring function used should anticipate rdkit_mols as input

            # Using both representations and a custom identifier
            >>> inchis = [Chem.MolToInchi(mol) for mol in mols]
            >>> scores = MS.score(smiles=smiles, mol_ids=inchis, rdkit_mols=mols)
        """
        # ----- Check and organise our molecular representations -----
        assert (
            smiles or len(molecular_inputs)
        ), "No molecular representations provided, please supply smiles or other representations as keyword arguments"
        if smiles:
            molecular_inputs["smiles"] = smiles
        assert all(
            isinstance(v, list) for v in molecular_inputs.values()
        ), "All molecular representations must be lists"
        _batch_size = [len(v) for v in molecular_inputs.values()]
        assert all(
            s == _batch_size[0] for s in _batch_size
        ), f"All molecular representations must of equal number but got {_batch_size}"
        _batch_size = _batch_size[0]

        if score_only:
            return self._score_only(
                step=step, flt=flt, **molecular_inputs
            )  

        # ----- Initialize some values -----
        batch_start = time.time()
        if step is not None:
            self.step = step
        else:
            self.step += 1
        logger.info(f"STEP {self.step}")
        logger.info(f"    Received: {_batch_size} molecular inputs")

        # ----- Parse smiles and initiate batch df -----
        # NOTE parse() set's self.batch_df["mol_ids"] as user provided, running idx, or canonical smiles
        self.parse(
            mol_ids=mol_ids,
            step=self.step,
            canonicalise_smiles=canonicalise_smiles,
            check_uniqueness=check_uniqueness,
            **molecular_inputs,
        )
        # Maybe update molecular inputs with canonical SMILES
        if smiles and canonicalise_smiles:
            molecular_inputs["smiles"] = self.batch_df["smiles"].tolist()
        logger.debug(f"    Pre-processed: {len(self.batch_df)} molecular inputs")

        # ----- Subset only the required molecular inputs to pass to scoring functions ----
        # Update batch indices
        _process_batch_idxs = list(range(_batch_size))
        if recalculate:
            # Pass all valid molecules to scoring functions
            if "valid" in self.batch_df.columns:
                if self.score_invalids:
                    pass
                else:
                    _process_batch_idxs = self.batch_df.loc[
                        self.batch_df.valid, "batch_idx"
                    ].tolist()
        else:
            # Pass all valid and unique molecules to scoring functions
            # If valid column, there should be unique column
            if ("valid" in self.batch_df.columns) and (
                "unique" in self.batch_df.columns
            ):
                if self.score_invalids:
                    _process_batch_idxs = self.batch_df.loc[
                        self.batch_df.unique,
                        "batch_idx",
                    ].tolist()
                else:
                    _process_batch_idxs = self.batch_df.loc[
                        self.batch_df.valid & self.batch_df.unique,
                        "batch_idx",
                    ].tolist()
            # But unique can exist without valid
            elif "unique" in self.batch_df.columns:
                _process_batch_idxs = self.batch_df.loc[
                    self.batch_df.unique, "batch_idx"
                ].tolist()
            else:
                pass
        _process_mol_ids = self.batch_df.loc[
            self.batch_df.batch_idx.isin(_process_batch_idxs), "mol_id"
            ].tolist()
        
        # If nothing to process then instead submit at least 1 (scoring function should handle invalid)       
        if len(_process_batch_idxs) == 0:
            logger.info("    No smiles to score so submitting first input")
            _process_batch_idxs = [0]
            _process_mol_ids = [self.batch_df.iloc[0]["mol_id"]]

        # Prepare file names and molecular inputs for scoring function
        _process_file_names = [f"{self.step}_{i}" for i in _process_batch_idxs]
        _process_molecular_inputs = {
            k: [m for i, m in enumerate(v) if i in _process_batch_idxs]
            for k, v in molecular_inputs.items()
        }
        logger.info(f"    Scoring: {len(_process_batch_idxs)} molecular inputs")

        # ----- Run scoring function -----
        scoring_start = time.time()
        self.run_scoring_functions(
            mol_ids=_process_mol_ids,
            batch_index=_process_batch_idxs,
            file_names=_process_file_names,
            **_process_molecular_inputs,
        )
        logger.debug(f"    Returned score for {len(self.results_df)} molecular inputs")
        logger.debug(f"    Scoring elapsed time: {time.time() - scoring_start:.02f}s")

        # ----- Append scoring results ----- 
        if isinstance(self.main_df, pd.core.frame.DataFrame) and not recalculate:
            self.concurrent_update()
        else:
            self.first_update()
        logger.debug(f"    Scores updated: {len(self.batch_df)} molecular inputs")

        # ----- Compute final fitness score -----
        self.update_maxmin(df=self.batch_df)
        self.batch_df = self.compute_score(df=self.batch_df)
        logger.debug(
            f"    Aggregate score calculated: {len(self.batch_df)} molecular inputs"
        )
        if self.diversity_filter is not None:
            self.batch_df = self.run_diversity_filter(self.batch_df)
            logger.info(
                f'    Passed diversity filter: {self.batch_df["passes_diversity_filter"].sum()} molecular inputs'
            )
            
        # ----- Add information of scoring time -----
        self.batch_df["score_time"] = time.time() - scoring_start

        # ----- Append batch df to main df if it exists, else initialise it -----
        if isinstance(self.main_df, pd.core.frame.DataFrame):
            self.main_df = pd.concat([self.main_df, self.batch_df], axis=0)
        else:
            self.main_df = self.batch_df.copy()
            
        # ----- Update current idx -----
        self.current_idx = self.main_df.index[-1] + 1

        # ----- Write out csv log for each iteration -----
        self.batch_df.to_csv(
            os.path.join(self.save_dir, "iterations", f"{self.step:06d}_scores.csv")
        )

        # ----- Update replay buffer -----
        if self.replay_size:
            self.replay_buffer.update(
                self.batch_df,
                endpoint_key="Score",
                using_DF=bool(self.diversity_filter),
                **molecular_inputs
            )

        # ----- Start GUI monitor -----
        if self.monitor_app is True:
            self.run_monitor()

        # ----- Fetch score ------
        if self.diversity_filter is not None:
            scores = self.batch_df.loc[
                :, "Score (reshaped)"
            ].tolist()
        else:
            scores = self.batch_df.loc[:, "Score"].tolist()
        if not flt:
            scores = np.array(scores, dtype=np.float32)
        logger.debug(f"    Returning: {len(scores)} scores")
        logger.info(f"    MolScore elapsed time: {time.time() - batch_start:.02f}s")

        # ----- Write out DF and RB memory intermittently -----
        if self.step % 5 == 0:
            if (self.diversity_filter is not None) and (
                self.diversity_filter not in ["Unique", "Occurrence"]
            ):
                self.diversity_filter.savetocsv(
                    os.path.join(self.save_dir, "scaffold_memory.csv")
                )
            if len(self.replay_buffer) > 0:
                self.replay_buffer.save(
                    os.path.join(self.save_dir, "replay_buffer.csv")
                )

        # ----- Clean up class -----
        self.evaluate_finished()
        self.batch_df = None
        self.results_df = None

        return scores

    def __call__(self, *args, **kwargs):
        """
        Directly calling MolScore is being deprecated, please use score() instead.
        """
        return self.score(*args, **kwargs)

    # ----- Additional methods only run if called directly -----
    def replay(self, n, molecule_key: str = 'smiles', augment: bool = False) -> Union[list, list]:
        """
        Sample n molecules from the replay buffer
        :param n: Number of molecules to sample
        :param molecule_key: What type of possible representation to return if present
        :param augment: Whether to augment the replay buffer by randomizing the smiles
        :return: List of SMILES and scores
        """
        return self.replay_buffer.sample(
            n=n, endpoint_key="Score", molecule_key=molecule_key, augment=augment
        )

    def compute_metrics(
        self,
        endpoints: list = None,
        thresholds: list = None,
        chemistry_filter_basic=True,
        budget=None,
        oracle_budget=None,
        n_jobs=1,
        reference_smiles=None,
        include=['Valid', 'Unique'],
        benchmark=None,
        recalculate=False,
    ):
        """
        Compute a suite of metrics

        :param endpoints: List of endpoints to compute metrics for e.g., 'amean', 'docking_score' etc.
        :param thresholds: List of thresholds to compute metrics for
        :param chemistry_filter_basic: Whether to apply basic chemistry filters
        :param budget: Molecule budget to compute metrics for
        :param oracle_budget: Oracle budget to compute metrics for
        :param n_jobs: Number of jobs to use for parallelisation
        :param reference_smiles: List of target smiles to compute metrics in reference to
        :param include: List of metrics to run
        :param benchmark: Benchmark to run (i.e., preset metrics)
        :param recalculate: Whether to recompute metrics if they've already been computed
        """
        if self.metrics and not recalculate:
            return self.metrics
        else:
            if endpoints is None:
                endpoints = ["Score"]
            else:
                assert all([ep in self.main_df.columns for ep in endpoints])
            if thresholds is None:
                thresholds = [0.0]
            SM = ScoreMetrics(
                scores=self.main_df,
                budget=budget,
                oracle_budget=oracle_budget,
                n_jobs=n_jobs,
                reference_smiles=reference_smiles,
                benchmark=benchmark,
            )
            results = SM.get_metrics(
                endpoints=endpoints,
                thresholds=thresholds,
                chemistry_filter_basic=chemistry_filter_basic,
                include=include,                
            )
            self.metrics = results
            return self.metrics


class MolScoreBenchmark:
    presets = PRESETS

    def __init__(
        self,
        model_name: str,
        output_dir: os.PathLike,
        budget: int = None,
        oracle_budget: int = None,
        replay_size: int = None,
        replay_purge: bool = True,
        score_invalids=False,
        add_benchmark_dir: bool = True,
        model_parameters: dict = {},
        benchmark: str = None,
        custom_benchmark: os.PathLike = None,
        custom_tasks: list = [],
        include: list = [],
        exclude: list = [],
        diversity_filter: str = None,
        **kwargs,
    ):
        """
        Run MolScore in benchmark mode, which will run MolScore on a set of tasks and benchmarks.
        :param model_name: Name of model to run
        :param output_dir: Directory to save results to
        :param budget: Molecule budget to run MolScore task for
        :param oracle_budget: Oracle budget to run MolScore task for
        :param replay_size: Size of replay buffer to store top scoring molecules
        :param replay_purge: Whether to purge replay buffer based on diversity filter
        :param score_invalids: Whether to force scoring of invalid molecules
        :param add_benchmark_dir: Whether to add benchmark directory to output directory
        :param model_parameters: Parameters of the model for record
        :param benchmark: Name of benchmark to run
        :param custom_benchmark: Path to custom benchmark directory
        :param custom_tasks: List of custom tasks to run
        :param include: List of tasks to only include
        :param exclude: List of tasks to exclude
        :param diversity_filter: Diversity filter to use, will replace any defined in configs
        """
        self.model_name = model_name
        self.model_parameters = model_parameters
        self.output_dir = output_dir
        self.benchmark = benchmark
        self.custom_benchmark = custom_benchmark
        self.custom_tasks = custom_tasks
        self.include = include
        self.exclude = exclude
        self.budget = budget
        self.oracle_budget = oracle_budget
        self.replay_size = replay_size
        self.replay_purge = replay_purge
        self.configs = []
        self.results = []
        self.score_paths = []
        self.score_invalids = score_invalids
        self.next = 0
        self.diversity_filter = diversity_filter
        
        # Confirm budget
        assert self.budget or self.oracle_budget, "Must specify a budget or oracle_budget"
        assert not (self.budget and self.oracle_budget), "Must specify either budget or oracle_budget, not both"

        # Process configs and tasks to run
        if self.benchmark:
            assert (
                self.benchmark in self.presets.keys()
            ), f"Benchmark {self.benchmark} not found in presets"
            for config in self.presets[self.benchmark].glob("*.json"):
                self.configs.append(str(config))

        if self.custom_benchmark:
            assert os.path.isdir(
                self.custom_benchmark
            ), f"Custom benchmark directory {self.custom_benchmark} not found"
            for config in os.listdir(self.custom_benchmark):
                if config.endswith(".json"):
                    self.configs.append(os.path.join(self.custom_benchmark, config))

        if self.custom_tasks:
            for task in os.listdir(self.custom_tasks):
                assert os.path.exists(task), f"Custom taks {task} not found"
                if task.endswith(".json"):
                    self.configs.append(os.path.join(self.custom_tasks, task))

        if self.include:
            exclude = []
            for config in self.configs:
                name = Path(config).stem
                if (name in self.include) or (name in self.include):
                    continue
                else:
                    exclude.append(config)
            for config in exclude:
                self.configs.remove(config)

        if self.exclude:
            exclude = []
            for config in self.configs:
                name = Path(config).stem
                if (name in self.exclude) or (name in self.exclude):
                    exclude.append(config)
            for config in exclude:
                self.configs.remove(config)

        # Check some configs are specified
        if not self.configs:
            raise ValueError(
                "No configs found to run, this could be due to include/exclude resulting in zero configs to run"
            )

        # Add benchmark directory
        if add_benchmark_dir:
            if self.benchmark:
                self.output_dir = os.path.join(
                    self.output_dir,
                    f"{benchmark}_{self.model_name}_{time.strftime('%Y_%m_%d-%H_%M_%S', time.localtime())}",
                )
            else:
                self.output_dir = os.path.join(
                    self.output_dir,
                    f"Benchmark_{self.model_name}_{time.strftime('%Y_%m_%d-%H_%M_%S', time.localtime())}",
                )
                
        os.makedirs(self.output_dir, exist_ok=True)
            
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Context Teardown"""
        self._write_results()
        if exc_type:
            print(f"Handled exception: {exc_value}")
        return False
    
    @contextmanager
    def _managed_task(self, config_path):
        MS = MolScore(
            model_name=self.model_name,
            task_config=config_path,
            output_dir=self.output_dir,
            budget=self.budget,
            oracle_budget=self.oracle_budget,
            termination_exit=False,
            replay_size=self.replay_size,
            replay_purge=self.replay_purge,
            score_invalids=self.score_invalids,
            diversity_filter=self.diversity_filter,
        )
        try:
            with MS as scoring_function:
                yield scoring_function
        finally:
            if MS.main_df is None:
                print(f"Skipping summary of {MS.cfg['task']} as no results found")
            else: 
                metrics = MS.compute_metrics(
                    budget=self.budget,
                    benchmark=self.benchmark,          
                )
                metrics.update(
                    {
                        "model_name": self.model_name,
                        "model_parameters": self.model_parameters,
                        "task": MS.cfg["task"],
                    }
                )
                self.results.append(metrics)
                self.score_paths.append(os.path.join(MS.save_dir, 'scores.csv'))
                    
    def __iter__(self):
        for config_path in self.configs:
            yield self._managed_task(config_path)

    def __len__(self):
        return len(self.configs)

    def summarize(
        self,
        endpoints=None,
        thresholds=None,
        chemistry_filter_basic=True,
        n_jobs=1,
        include=['Valid', 'Unique'],
        reference_smiles=None,
        overwrite=True,
    ):
        """
        Manually summarize benchmark and write results
        """
        print("Calculating summary metrics")
        if endpoints is None:
            endpoints = ['Score']
        self.results = []
        # Compute results
        for score_path in self.score_paths:
            if not os.path.exists(score_path):
                print(f"Skipping summary of {score_path} as no results found")
                continue
            scores = pd.read_csv(score_path, index_col=0)
            
            task_name = scores["task"].iloc[0]
            
            SM = ScoreMetrics(
                scores=scores,
                budget=self.budget,
                oracle_budget=self.oracle_budget,
                n_jobs=n_jobs,
                reference_smiles=reference_smiles,
                benchmark=self.benchmark,
            )
            metrics = SM.get_metrics(
                endpoints=endpoints,
                thresholds=thresholds,
                chemistry_filter_basic=chemistry_filter_basic,
                include=include,                
            )
            metrics["task"] = task_name
            self.results.append(metrics)
        self._write_results(overwrite=overwrite)
        return self.results
    
    def _write_results(self, overwrite=False):
        if not os.path.exists(os.path.join(self.output_dir, "results.csv")) or overwrite:
            # Save results
            pd.DataFrame(self.results).to_csv(
                os.path.join(self.output_dir, "results.csv"), index=False
            )
            # Print results
            print(f"Preview of Results:\n{(pd.DataFrame(self.results))}")


class MolScoreCurriculum(MolScore):
    presets = {}

    def __init__(
        self,
        model_name: str,
        output_dir: os.PathLike,
        run_name: str = None,
        model_parameters: dict = {},
        budget: int = None,
        oracle_budget: int = None,
        termination_threshold: float = None,
        termination_patience: int = None,
        replay_size: int = None,
        replay_purge: bool = True,
        score_invalids: bool = False,
        reset_replay_buffer: bool = False,
        benchmark: str = None,
        custom_benchmark: os.PathLike = None,
        custom_tasks: list = [],
        include: list = [],
        exclude: list = [],
        **kwargs,
    ):
        """
        Run MolScore in curriculum mode, which will change MolScore tasks based on thresholds / steps reached.
        WARNING, sorted by config name! E.g., 0_Albuterol.json, 1_QED.json ...
        :param model_name: Name of model to run
        :param output_dir: Directory to save results to
        :param run_name: Name of Curriculum run if specified
        :param model_parameters: Parameters of the model for record
        :param budget: Molecule budget to run each MolScore task for
        :param oracle_budget: Oracle budget to run each MolScore task for
        :param termination_threshold: Threshold to terminate MolScore task
        :param termination_patience: Number of steps to wait before terminating MolScore task
        :param replay_size: Size of replay buffer to store top scoring molecules
        :param replay_purge: Whether to purge replay buffer based on diversity filter
        :param score_invalids: Whether to force scoring of invalid molecules
        :param reset_replay_buffer: Whether to reset replay buffer between tasks
        :param benchmark: Name of benchmark to run
        :param custom_benchmark: Path to custom benchmark directory containing configs
        :param custom_tasks: List of custom tasks to run
        :param include: List of tasks to only include
        :param exclude: List of tasks to exclude
        """
        self.model_name = model_name
        self.model_parameters = model_parameters
        self.budget = budget
        self.oracle_budget = oracle_budget
        self.termination_threshold = termination_threshold
        self.termination_patience = termination_patience
        self.replay_size = replay_size
        self.replay_purge = replay_purge
        self.reset_replay_buffer = reset_replay_buffer
        # If any termination criteria is set here, this is propogated to all tasks
        self.reset_from_config = not any(
            [budget, oracle_budget, termination_threshold, termination_patience]
        )
        self.output_dir = output_dir
        self.benchmark = benchmark
        self.custom_benchmark = custom_benchmark
        self.custom_tasks = custom_tasks
        self.include = include
        self.exclude = exclude
        self.configs = []
        self.score_invalids = score_invalids
        
        # Confirm budget
        assert not (self.budget and self.oracle_budget), "Must specify either budget or oracle_budget, not both"

        # Process configs and tasks to run
        if self.benchmark:
            assert (
                self.benchmark in self.presets.keys()
            ), f"Preset {self.benchmark} not found in presets"
            for config in self.presets[self.benchmark].glob("*.json"):
                self.configs.append(str(config))

        if self.custom_benchmark:
            assert os.path.isdir(
                self.custom_benchmark
            ), f"Custom benchmark directory {self.custom_benchmark} not found"
            for config in os.listdir(self.custom_benchmark):
                if config.endswith(".json"):
                    self.configs.append(os.path.join(self.custom_benchmark, config))

        if self.custom_tasks:
            for task in os.listdir(self.custom_tasks):
                assert os.path.exists(task), f"Custom taks {task} not found"
                if task.endswith(".json"):
                    self.configs.append(os.path.join(self.custom_tasks, task))

        if self.include:
            exclude = []
            for config in self.configs:
                name = Path(config).stem
                if (name in self.include) or (name in self.include):
                    continue
                else:
                    exclude.append(config)
            for config in exclude:
                self.configs.remove(config)

        if self.exclude:
            exclude = []
            for config in self.configs:
                name = Path(config).stem
                if (name in self.exclude) or (name in self.exclude):
                    exclude.append(config)
            for config in exclude:
                self.configs.remove(config)

        # Check some configs are specified
        if not self.configs:
            raise ValueError(
                "No configs found to run, this could be due to include/exclude resulting in zero configs to run"
            )

        # Order them by alphabetical order
        assert all(
            [re.search("^([0-9]*)", os.path.basename(c)).group() for c in self.configs]
        ), "Config files must be prefixed with a number"
        self.configs.sort(
            key=lambda x: int(re.search("^([0-9]*)", os.path.basename(x)).group())
        )

        super().__init__(
            model_name=self.model_name,
            task_config=self.configs.pop(0),
            output_dir=self.output_dir,
            run_name=run_name,
            budget=self.budget,
            oracle_budget=self.oracle_budget,
            termination_threshold=self.termination_threshold,
            termination_patience=self.termination_patience,
            termination_exit=False,
            replay_size=self.replay_size,
            replay_purge=self.replay_purge,
            score_invalids=self.score_invalids,
        )

    def score(self, *args, **kwargs):
        output = super().score(*args, **kwargs)
        if self.finished:
            # Move on to the next task if available
            if self.configs:
                logger.info("Moving on to the next task ...")
                self._set_objective(
                    task_config=self.configs.pop(0),
                    reset_termination_criteria=self.reset_from_config,
                    reset_replay_buffer=self.reset_replay_buffer,
                )
                self.finished = False
                self.termination_counter = 0
        return output

    def __call__(self, *args, **kwargs):
        self.score(*args, **kwargs)