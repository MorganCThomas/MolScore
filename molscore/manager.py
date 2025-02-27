import atexit
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from typing import Union

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
    }


class MolScore:
    """
    Central manager class that, when called, takes in a list of SMILES and returns respective scores.
    """
    presets = PRESETS

    preset_tasks = {
        k:[p.name.strip(".json") for p in v.glob("*.json")] 
        for k, v in presets.items()
        }
    
    @staticmethod
    def load_config(task_config):
        assert os.path.exists(
            task_config
        ), f"Configuration file {task_config} doesn't exist"
        with open(task_config, "r") as f:
            configs = f.read().replace("\r", "").replace("\n", "").replace("\t", "")
        return json.loads(configs)

    def __init__(
        self,
        model_name: str,
        task_config: Union[str, os.PathLike],
        output_dir: str = None,
        add_run_dir: bool = True,
        run_name: str = None,
        budget: int = None,
        termination_threshold: int = None,
        termination_patience: int = None,
        termination_exit: bool = False,
        score_invalids: bool = False,
        replay_size: int = None,
        replay_purge: bool = True,
        **kwargs,
    ):
        """
        :param model_name: Name of generative model, used for file naming and documentation
        :param task_config: Path to task config file, or a preset name e.g., GuacaMol:Albuterol_similarity
        :param output_dir: Overwrites the output directory specified in the task config file
        :param add_run_dir: Adds a run directory within the output directory
        :param run_name: Override the run name with a custom name, otherwise taken from 'task' in the config
        :param budget: Budget number of molecules to run MolScore task for until molscore.finished is True
        :param termination_threshold: Threshold for early stopping based on the score
        :param termination_patience: Number of steps with no improvement, or that a termination_threshold has been reached for
        :param termination_exit: Exit on termination of objective
        :param replay_size: Maximum size of the replay buffer
        :param replay_purge: Whether to purge the replay buffer, i.e., only allow molecules that pass the diversity filter
        """
        # Load in configuration file (json)
        if task_config.endswith(".json"):
            self.cfg = self.load_config(task_config)
        else:
            assert ":" in task_config, "Preset task must be in format 'category:task'"
            cat, task = task_config.split(":", maxsplit=1)
            assert cat in self.presets.keys(), f"Preset category {cat} not found"
            task_config = self.presets[cat] / f"{task}.json"
            assert task_config.exists(), f"Preset task {task} not found in {cat}"
            self.cfg = self.load_config(task_config)

        # Here are attributes used
        self.model_name = model_name
        self.step = 0
        self.budget = budget
        self.termination_threshold = termination_threshold
        self.termination_patience = termination_patience
        reset_termination_criteria = not any(
            [budget, termination_threshold, termination_patience]
        )
        self.termination_counter = 0
        self.termination_exit = termination_exit
        self.score_invalids = score_invalids
        self.replay_size = replay_size
        self.replay_purge = replay_purge
        self.replay_buffer = utils.ReplayBuffer(size=replay_size, purge=replay_purge)
        self.finished = False
        self.init_time = time.time()
        self.results_df = None # TODO make empty df ... 
        self.batch_df = None
        self.exists_df = pd.DataFrame()
        self.main_df = None
        self.monitor_app = None
        self.diversity_filter = None
        self.call2score_warning = True
        self.metrics = None
        self.logged_parameters = {}  # Extra parameters to write out in scores.csv for comparative purposes
        self.temp_parameters = {} # Temp parameters to write out each iteration in scores.csv for comparative purposes

        # Setup save directory
        if not run_name:
            run_name = self.cfg["task"].replace(" ", "_")
        self.run_name = "_".join(
            [
                time.strftime("%Y_%m_%d", time.localtime()),
                self.model_name,
                run_name,
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
                dtype={"Unnamed: 0": "int64", "valid": object, "unique": object},
            )
            logger.debug(self.main_df.head())
            # Update step
            self.step = max(self.main_df["step"])
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

        # Registor write_scores and kill_monitor at close
        atexit.register(self.write_scores)
        atexit.register(self.kill_monitor)
        logger.info("MolScore initiated")

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

    def parse_smiles(self, smiles: list, step: int, canonicalize: bool = True):
        """
        Create batch_df object from initial list of SMILES and calculate validity and
        intra-batch uniqueness

        :param smiles: List of smiles taken from generative model
        :param step: current generative model step
        :param canonicalize: Whether to canonicalize smiles before scoring
        """
        # Initialize df for batch
        self.batch_df = pd.DataFrame(index=range(len(smiles)))

        # Parse smiles
        parsed_smiles = []
        valid = []
        batch_idx = []
        for i, smi in enumerate(smiles):
            try:
                can_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
                if canonicalize:
                    parsed_smiles.append(can_smi)
                else:
                    parsed_smiles.append(smi)
                valid.append("true")
            except TypeError:
                try:
                    mol = Chem.MolFromSmiles(smi)
                    Chem.SanitizeMol(mol)  # Try to catch invalid molecules and sanitize
                    can_smi = Chem.MolToSmiles(mol)
                    if canonicalize:
                        parsed_smiles.append(can_smi)
                    else:
                        parsed_smiles.append(smi)
                    valid.append("sanitized")
                except Exception:
                    parsed_smiles.append(smi)
                    valid.append("false")
            batch_idx.append(i)

        self.batch_df["model"] = self.model_name.replace(" ", "_")
        self.batch_df["task"] = self.cfg["task"].replace(" ", "_")
        self.batch_df["step"] = step
        self.batch_df["batch_idx"] = batch_idx
        self.batch_df["absolute_time"] = time.time() - self.init_time
        self.batch_df["smiles"] = parsed_smiles
        self.batch_df["valid"] = valid
        self.batch_df["valid_score"] = [1 if v == "true" else 0 for v in valid]  ##

        # Check for duplicates
        duplicated = self.batch_df.smiles.duplicated().tolist()
        unique = [
            str(not b).lower() for b in duplicated
        ]  # Reverse true/false i.e. unique as oppose to duplicated
        self.batch_df["unique"] = unique

        # Count previous occurrences
        occurrences = [
            self.batch_df.smiles.iloc[:i][
                self.batch_df.smiles.iloc[:i] == self.batch_df.smiles.iloc[i]
            ].count()
            for i in range(len(self.batch_df))
        ]
        self.batch_df["occurrences"] = occurrences

        number_invalid = len(self.batch_df.loc[self.batch_df.valid == "false", :])
        logger.debug(f"    Invalid molecules: {number_invalid}")
        return self

    def check_uniqueness(self):
        """
        Check batch_df smiles against main_df of any previously sampled smiles
        """

        # Pull duplicated smiles from the main df
        self.exists_df = self.main_df[
            self.main_df.smiles.isin(self.batch_df.smiles.tolist())
        ]

        # Update unique and occurrence columns
        if len(self.exists_df) > 0:
            for smi in self.batch_df.smiles.unique():
                tdf = self.exists_df[self.exists_df.smiles == smi]
                if len(tdf) > 0:
                    self.batch_df.loc[self.batch_df.smiles == smi, "unique"] = "false"
                    self.batch_df.loc[self.batch_df.smiles == smi, "occurrences"] += (
                        len(tdf)
                    )
        return self

    def run_scoring_functions(
        self, smiles: list, file_names: list, additional_formats: dict = None
    ):
        """
        Calculate respective scoring function scores for a list of unique smiles
         (with file names for logging if necessary).

        :param smiles: A list of valid smiles, preferably without duplicated or known scores
        :param file_names: A corresponding list of file prefixes for tracking - format={step}_{batch_idx}
        :return: self.results (a list of dictionaries with smiles and resulting scores)
        """
        self.results_df = pd.DataFrame(smiles, columns=["smiles"])
        for function in self.scoring_functions:
            results = function(
                smiles=smiles,
                directory=self.save_dir,
                file_names=file_names,
                additional_formats=additional_formats,
            )
            results_df = pd.DataFrame(results)

            self.results_df = self.results_df.merge(
                results_df, on="smiles", how="outer", sort=False
            )

        # Drop any duplicates in results
        self.results_df = self.results_df.drop_duplicates(subset="smiles")
        return self

    def first_update(self):
        """
        Append calculated scoring function values to batch dataframe. Only used for the first step/batch.
        """
        logger.debug("    Merging results to batch df")
        self.batch_df = self.batch_df.merge(
            self.results_df, on="smiles", how="left", sort=False
        )
        self.batch_df.fillna(0.0, inplace=True)
        return self

    def concurrent_update(self):
        """
        Append calculated scoring function values to batch dataframe while looking up duplicated entries to avoid
        re-calculating.

        :return:
        """
        # Grab data for pre-existing smiles
        if len(self.exists_df) > 0:
            self.exists_df = self.exists_df.drop_duplicates(subset="smiles")
            self.exists_df = self.exists_df.loc[:, self.results_df.columns]
            # Check no duplicated values in exists and results df
            dup_idx = self.exists_df.loc[
                self.exists_df.smiles.isin(self.results_df.smiles), :
            ].index.tolist()
            if len(dup_idx) > 0:
                self.exists_df.drop(index=dup_idx, inplace=True)
            # Append to results, assuming no duplicates in results_df...
            self.results_df = pd.concat(
                [self.results_df, self.exists_df], axis=0, ignore_index=True, sort=False
            )

        # Merge with batch_df
        logger.debug("    Merging results to batch df")
        self.batch_df = self.batch_df.merge(
            self.results_df, on="smiles", how="left", sort=False
        )
        self.batch_df.fillna(0.0, inplace=True)
        return self

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
        return self

    def compute_score(self, df):
        """
        Compute the final score i.e. combination of which metrics according to which method.
        """
        mpo_columns = {"names": [], "weights": []}
        filter_columns = {"names": []}
        # Iterate through specified metrics and apply modifier
        transformed_columns = {}
        for metric in self.cfg["scoring"]["metrics"]:
            mod_name = f"{metric['modifier']}_{metric['name']}"
            # NEW filter_columns else mpo_columns
            if metric.get("filter", False):
                filter_columns["names"].append(mod_name)
            else:
                mpo_columns["names"].append(mod_name)
                mpo_columns["weights"].append(metric["weight"])

            for mod in self.modifier_functions:
                if metric["modifier"] == mod.__name__:
                    modifier = mod

            # Check the modifier function exists, and the metric can be found in the dataframe
            assert any(
                [metric["modifier"] == mod.__name__ for mod in self.modifier_functions]
            ), f"Score modifier {metric['modifier']} not found"
            try:
                assert (
                    metric["name"] in df.columns
                ), f"Specified metric {metric['name']} not found in dataframe"
            except AssertionError as e:
                self._write_temp_state(step=self.step)
                raise e

            transformed_columns[mod_name] = (
                df.loc[:, metric["name"]]
                .apply(lambda x: modifier(x, **metric["parameters"]))
                .rename(mod_name)
            )
        df = pd.concat([df] + list(transformed_columns.values()), axis=1)
        # Double check we have no NaN or 0 values (necessary for geometric mean) for mpo columns
        df.loc[:, mpo_columns["names"]].fillna(1e-6, inplace=True)
        df[mpo_columns["names"]] = df[mpo_columns["names"]].apply(
            lambda x: [1e-6 if y < 1e-6 else y for y in x]
        )

        # Compute final score
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

        # NEW Add filter metrics
        df["filter"] = df.loc[:, filter_columns["names"]].apply(
            lambda x: np.prod(x), axis=1, raw=True
        )
        df[self.cfg["scoring"]["method"]] = (
            df[self.cfg["scoring"]["method"]] * df["filter"]
        )

        return df

    def run_diversity_filter(self, df):
        if self.diversity_filter == "Unique":
            df[f"filtered_{self.cfg['scoring']['method']}"] = [
                s if u == "true" else 0.0
                for u, s in zip(df["unique"], df[self.cfg["scoring"]["method"]])
            ]
            df["passes_diversity_filter"] = [
                True if float(a) == float(b) else False
                for b, a in zip(
                    df[self.cfg["scoring"]["method"]],
                    df[f"filtered_{self.cfg['scoring']['method']}"],
                )
            ]
            df.fillna(1e-6)

        elif self.diversity_filter == "Occurrence":
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
            df.fillna(1e-6)
        return df

    def log_parameters(self, parameters: dict):
        self.logged_parameters.update(parameters)
        return self

    def write_scores(self):
        """
        Write final dataframe to file.
        """
        if self.main_df is not None:
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

        if (self.diversity_filter is not None) and (
            isinstance(self.diversity_filter, scaffold_memory.ScaffoldMemory)
        ):
            self.diversity_filter.savetocsv(
                os.path.join(self.save_dir, "scaffold_memory.csv")
            )
        if len(self.replay_buffer) > 0:
            self.replay_buffer.save(os.path.join(self.save_dir, "replay_buffer.csv"))

        self.fh.close()

        return self

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
            self.exists_df.to_csv(os.path.join(self.save_dir, f"exists_df_{step}.csv"))
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
        return self

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

        # Based on budget
        if self.budget and (len(task_df) >= self.budget):
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

    def score_only(self, smiles: list, step: int = None, flt: bool = False):
        batch_start = time.time()
        if step is not None:
            self.step = step
        logger.info(f"   Scoring: {len(smiles)} SMILES")
        run_smiles = list(set(smiles))
        file_names = [f"{self.step}_{i}" for i, smi in enumerate(run_smiles)]
        self.run_scoring_functions(smiles=run_smiles, file_names=file_names)
        logger.info(
            f"    Score returned for {len(self.results_df)} SMILES in {time.time() - batch_start:.02f}s"
        )
        self.update_maxmin(self.results_df)
        self.results_df = self.compute_score(self.results_df)
        if self.diversity_filter is not None:
            self.results_df = self.run_diversity_filter(self.results_df)
            score_col = f"filtered_{self.cfg['scoring']['method']}"
        else:
            score_col = self.cfg["scoring"]["method"]

        scores = [
            float(self.results_df.loc[self.results_df.smiles == smi, score_col])
            for smi in smiles
        ]
        if not flt:
            scores = np.array(scores, dtype=np.float32)
        logger.info(
            f"    Score returned for {len(self.results_df)} SMILES in {time.time() - batch_start:.02f}s"
        )

        # Clean up class
        self.batch_df = None
        self.exists_df = pd.DataFrame()
        self.results_df = None
        return scores

    def score(
        self,
        smiles: list,
        step: int = None,
        flt: bool = False,
        canonicalize: bool = True,
        recalculate: bool = False,
        check_uniqueness: bool = True,
        score_only: bool = False,
        additional_formats: dict = None,
        additional_keys: dict = None,
        **kwargs,
    ):
        """
        Calling this method will result in the primary function of scoring smiles and logging data in
         an automated fashion, and returning output values.

        :param smiles: A list of smiles for scoring.
        :param step: Step of generative model for logging, and indexing. This could equally be iterations/epochs etc.
        :param flt: Whether to return a list of floats (default False i.e. return np.array of type np.float32)
        :param canonicalize: Whether to canonicalize smiles before scoring
        :param recalculate: Whether to recalculate scores for duplicated values,
         in case scoring function may be somewhat stochastic.
         (default False i.e. use existing scores for duplicated molecules and renormalize/penalize if necessary)
        :param check_uniqueness: Whether to check for uniqueness against cache of molecules in main_df
        :param score_only: Whether to log molecule data or simply score and return
        :param additional_formats: Additional formats to be passed to scoring functions
        :param additional_keys: Additional keys to store in the scores.csv file as new columns
        :return: Scores (either float list or np.array)
        """
        if score_only:
            return self.score_only(smiles=smiles, step=step, flt=flt)

        # Set some values
        batch_start = time.time()
        if step is not None:
            self.step = step
        else:
            self.step += 1
        logger.info(f"STEP {self.step}")
        logger.info(f"    Received: {len(smiles)} SMILES")

        # Parse smiles and initiate batch df
        self.parse_smiles(smiles=smiles, step=self.step, canonicalize=canonicalize)
        logger.debug(f"    Pre-processed: {len(self.batch_df)} SMILES")
        logger.info(f'    Invalids found: {(self.batch_df.valid == "false").sum()}')

        # Add temp parameters or additional keys to batch_df
        if self.temp_parameters:
            for k, v in self.temp_parameters.items():
                if k in self.batch_df.columns:
                    logger.warning(
                        f"    Overwriting existing column {k} with temp parameter"
                    )
                self.batch_df[k] = v
            self.temp_parameters = {}
        if additional_keys is not None:
            for k, v in additional_keys.items():
                if k in self.batch_df.columns:
                    logger.warning(
                        f"    Overwriting existing column {k} with additional key"
                    )
                self.batch_df[k] = v

        # If a main df exists check if some molecules have already been sampled
        if isinstance(self.main_df, pd.core.frame.DataFrame) and check_uniqueness:
            self.check_uniqueness()
            logger.debug(f"    Uniqueness updated: {len(self.batch_df)} SMILES")
        logger.info(
            f'    Duplicates found: {(self.batch_df.unique == "false").sum()} SMILES'
        )

        # Subset only unique and valid smiles
        if self.score_invalids:
            if recalculate:
                smiles_to_process = self.batch_df.loc[:, "smiles"].tolist()
                smiles_to_process_index = self.batch_df.loc[:, "batch_idx"].tolist()
            else:
                smiles_to_process = self.batch_df.loc[
                    (self.batch_df.unique == "true"),
                    "smiles",
                ].tolist()
                smiles_to_process_index = self.batch_df.loc[
                    (self.batch_df.unique == "true"),
                    "batch_idx",
                ].tolist()
        else:
            if recalculate:
                smiles_to_process = self.batch_df.loc[
                    self.batch_df.valid.isin(["true", "sanitized"]), "smiles"
                ].tolist()
                smiles_to_process_index = self.batch_df.loc[
                    self.batch_df.valid.isin(["true", "sanitized"]), "batch_idx"
                ].tolist()
            else:
                smiles_to_process = self.batch_df.loc[
                    (self.batch_df.valid.isin(["true", "sanitized"]))
                    & (self.batch_df.unique == "true"),
                    "smiles",
                ].tolist()
                smiles_to_process_index = self.batch_df.loc[
                    (self.batch_df.valid.isin(["true", "sanitized"]))
                    & (self.batch_df.unique == "true"),
                    "batch_idx",
                ].tolist()
        if len(smiles_to_process) == 0:
            # If no smiles to process then instead submit 10 (scoring function should handle invalid)
            logger.info("    No smiles to score so submitting first 10 SMILES")
            smiles_to_process = self.batch_df.loc[:9, "smiles"].tolist()
            smiles_to_process_index = self.batch_df.loc[:9, "batch_idx"].tolist()

        assert len(smiles_to_process) == len(smiles_to_process_index)
        file_names = [f"{self.step}_{i}" for i in smiles_to_process_index]
        logger.info(f"    Scoring: {len(smiles_to_process)} SMILES")

        # If additional formats are specified, then index and run these too
        if additional_formats is not None:
            additional_formats = {
                k: [m for i, m in enumerate(v) if i in smiles_to_process_index]
                for k, v in additional_formats.items()
            }

        # Run scoring function
        scoring_start = time.time()
        self.run_scoring_functions(
            smiles=smiles_to_process,
            file_names=file_names,
            additional_formats=additional_formats,
        )
        logger.debug(f"    Returned score for {len(self.results_df)} SMILES")
        logger.debug(f"    Scoring elapsed time: {time.time() - scoring_start:.02f}s")

        # Append scoring results
        if isinstance(self.main_df, pd.core.frame.DataFrame) and not recalculate:
            self.concurrent_update()
        else:
            self.first_update()
        logger.debug(f"    Scores updated: {len(self.batch_df)} SMILES")

        # Compute average / score
        self.update_maxmin(df=self.batch_df)
        self.batch_df = self.compute_score(df=self.batch_df)
        logger.debug(f"    Aggregate score calculated: {len(self.batch_df)} SMILES")
        if self.diversity_filter is not None:
            self.batch_df = self.run_diversity_filter(self.batch_df)
            logger.info(
                f'    Passed diversity filter: {self.batch_df["passes_diversity_filter"].sum()} SMILES'
            )

        # Add information of scoring time
        self.batch_df["score_time"] = time.time() - scoring_start

        # Append batch df to main df if it exists, else initialise it.
        if isinstance(self.main_df, pd.core.frame.DataFrame):
            # update indexing based on most recent index
            self.batch_df.index = self.batch_df.index + self.main_df.index[-1] + 1
            self.main_df = pd.concat([self.main_df, self.batch_df], axis=0)
        else:
            self.main_df = self.batch_df.copy()

        # Write out csv log for each iteration
        self.batch_df.to_csv(
            os.path.join(self.save_dir, "iterations", f"{self.step:06d}_scores.csv")
        )

        # Update replay buffer
        if self.replay_size:
            self.replay_buffer.update(
                self.batch_df,
                endpoint=self.cfg["scoring"]["method"],
                using_DF=bool(self.diversity_filter),
            )

        # Start dash_utils monitor to track iteration files once first one is written!
        if self.monitor_app is True:
            self.run_monitor()

        # Fetch score
        if self.diversity_filter is not None:
            scores = self.batch_df.loc[
                :, f"filtered_{self.cfg['scoring']['method']}"
            ].tolist()
        else:
            scores = self.batch_df.loc[:, self.cfg["scoring"]["method"]].tolist()
        if not flt:
            scores = np.array(scores, dtype=np.float32)
        logger.debug(f"    Returning: {len(scores)} scores")
        logger.info(f"    MolScore elapsed time: {time.time() - batch_start:.02f}s")

        # Write out memory intermittently
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

        # Clean up class
        self.evaluate_finished()
        self.batch_df = None
        self.exists_df = pd.DataFrame()
        self.results_df = None

        return scores

    def __call__(
        self,
        smiles: list,
        step: int = None,
        flt: bool = False,
        canonicalize: bool = True,
        recalculate: bool = False,
        score_only: bool = False,
        additional_formats: dict = None,
        additional_keys: dict = None,
    ):
        """
        Calling MolScore will result in the primary function of scoring smiles and logging data in
         an automated fashion, and returning output values.

        :param smiles: A list of smiles for scoring.
        :param step: Step of generative model for logging, and indexing. This could equally be iterations/epochs etc.
        :param flt: Whether to return a list of floats (default False i.e. return np.array of type np.float32)
        :param canonicalize: Whether to canonicalize smiles before scoring
        :param recalculate: Whether to recalculate scores for duplicated values,
         in case scoring function may be somewhat stochastic.
          (default False i.e. use existing scores for duplicated molecules)
        :param score_only: Whether to log molecule data or simply score and return
        :param additional_formats: Additional formats to be passed to scoring functions
        :param additional_keys: Additional keys to store in the scores.csv file as new columns
        :return: Scores (either float list or np.array)
        """
        return self.score(
            smiles=smiles,
            step=step,
            flt=flt,
            canonicalize=canonicalize,
            recalculate=recalculate,
            score_only=score_only,
            additional_formats=additional_formats,
            additional_keys=additional_keys,
        )

    # ----- Additional methods only run if called directly -----
    def replay(self, n, augment: bool = False) -> Union[list, list]:
        """
        Sample n molecules from the replay buffer
        :param n: Number of molecules to sample
        :param augment: Whether to augment the replay buffer by randomizing the smiles
        :return: List of SMILES and scores
        """
        return self.replay_buffer.sample(
            n=n, endpoint=self.cfg["scoring"]["method"], augment=augment
        )

    def compute_metrics(
        self,
        endpoints: list = None,
        thresholds: list = None,
        chemistry_filter_basic=True,
        budget=None,
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
        :budget: Budget to compute metrics for
        :n_jobs: Number of jobs to use for parallelisation
        :reference_smiles: List of target smiles to compute metrics in reference to
        :recalculate: Whether to recompute metrics
        """
        if self.metrics and not recalculate:
            return self.metrics
        else:
            if endpoints is None:
                endpoints = [self.cfg["scoring"]["method"]]
            else:
                assert all([ep in self.main_df.columns for ep in endpoints])
            if thresholds is None:
                thresholds = [0.0]
            SM = ScoreMetrics(
                scores=self.main_df,
                budget=budget,
                n_jobs=n_jobs,
                reference_smiles=reference_smiles,
                benchmark=benchmark,
            )
            results = SM.get_metrics(
                endpoints=endpoints,
                thresholds=thresholds,
                chemistry_filter_basic=chemistry_filter_basic,
                include=include
            )
            # Change the name of the default score to "Score"
            results = {
                k.replace(self.cfg["scoring"]["method"], "Score"): v
                for k, v in results.items()
            }
            self.metrics = results
            return self.metrics


class MolScoreBenchmark:
    presets = PRESETS

    def __init__(
        self,
        model_name: str,
        output_dir: os.PathLike,
        budget: int,
        replay_size: int = None,
        replay_purge: bool = True,
        add_benchmark_dir: bool = True,
        model_parameters: dict = {},
        benchmark: str = None,
        custom_benchmark: os.PathLike = None,
        custom_tasks: list = [],
        include: list = [],
        exclude: list = [],
        score_invalids=False,
        **kwargs,
    ):
        """
        Run MolScore in benchmark mode, which will run MolScore on a set of tasks and benchmarks.
        :param model_name: Name of model to run
        :param output_dir: Directory to save results to
        :param budget: Budget number of molecules to run MolScore task for
        :param replay_size: Size of replay buffer to store top scoring molecules
        :param replay_purge: Whether to purge replay buffer based on diversity filter
        :param add_benchmark_dir: Whether to add benchmark directory to output directory
        :param model_parameters: Parameters of the model for record
        :param benchmark: Name of benchmark to run
        :param custom_benchmark: Path to custom benchmark directory
        :param custom_tasks: List of custom tasks to run
        :param include: List of tasks to only include
        :param exclude: List of tasks to exclude
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
        self.replay_size = replay_size
        self.replay_purge = replay_purge
        self.configs = []
        self.results = []
        self.score_invalids = score_invalids
        self.next = 0
        atexit.register(self._summarize)

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
                name = os.path.basename(config)
                if (name in self.include) or (name.strip(".json") in self.include):
                    continue
                else:
                    exclude.append(config)
            for config in exclude:
                self.configs.remove(config)

        if self.exclude:
            exclude = []
            for config in self.configs:
                name = os.path.basename(config)
                if (name in self.exclude) or (name.strip(".json") in self.exclude):
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
            self.output_dir = os.path.join(
                self.output_dir,
                f"{time.strftime('%Y_%m_%d', time.localtime())}_{self.model_name}_benchmark{time.strftime('_%H_%M_%S', time.localtime())}",
            )

    def __iter__(self):
        for config_path in self.configs:
            # Instantiate MolScore
            MS = MolScore(
                model_name=self.model_name,
                task_config=config_path,
                output_dir=self.output_dir,
                budget=self.budget,
                termination_exit=False,
                score_invalids=self.score_invalids,
                replay_size=self.replay_size,
                replay_purge=self.replay_purge,
            )
            self.results.append(MS)
            yield MS

    def __len__(self):
        return len(self.configs)

    def summarize(
        self,
        endpoints=None,
        thresholds=None,
        chemistry_filter_basic=True,
        n_jobs=1,
        reference_smiles=None,
        include=['Valid', 'Unique'],
    ):
        """
        For each result, compute metrics and summary of all results
        """
        print("Calculating summary metrics")
        results = []
        # Compute results
        for MS in self.results:
            if MS.main_df is None:
                print(f"Skipping summary of {MS.cfg['task']} as no results found")
                continue
            metrics = MS.compute_metrics(
                endpoints=endpoints,
                thresholds=thresholds,
                chemistry_filter_basic=chemistry_filter_basic,
                budget=self.budget,
                n_jobs=n_jobs,
                reference_smiles=reference_smiles,
                benchmark=self.benchmark,
                include=include
            )
            metrics.update(
                {
                    "model_name": self.model_name,
                    "model_parameters": self.model_parameters,
                    "task": MS.cfg["task"],
                }
            )
            results.append(metrics)
        # Save results
        pd.DataFrame(results).to_csv(
            os.path.join(self.output_dir, "results.csv"), index=False
        )
        # Print results
        print(f"Preview of Results:\n{pd.DataFrame(results)}")
        return results

    def _summarize(self):
        """
        If results aren't saved, save just incase
        """
        if not os.path.exists(os.path.join(self.output_dir, "results.csv")):
            results = self.summarize()
            pd.DataFrame(results).to_csv(
                os.path.join(self.output_dir, "results.csv"), index=False
            )


class MolScoreCurriculum(MolScore):
    presets = {}

    def __init__(
        self,
        model_name: str,
        output_dir: os.PathLike,
        run_name: str = None,
        model_parameters: dict = {},
        budget: int = None,
        termination_threshold: float = None,
        termination_patience: int = None,
        replay_size: int = None,
        replay_purge: bool = True,
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
        :param budget: Budget number of molecules to run each MolScore task for
        :param termination_threshold: Threshold to terminate MolScore task
        :param termination_patience: Number of steps to wait before terminating MolScore task
        :param replay_size: Size of replay buffer to store top scoring molecules
        :param replay_purge: Whether to purge replay buffer based on diversity filter
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
        self.termination_threshold = termination_threshold
        self.termination_patience = termination_patience
        self.replay_size = replay_size
        self.replay_purge = replay_purge
        self.reset_replay_buffer = reset_replay_buffer
        # If any termination criteria is set here, this is propogated to all tasks
        self.reset_from_config = not any(
            [budget, termination_threshold, termination_patience]
        )
        self.output_dir = output_dir
        self.benchmark = benchmark
        self.custom_benchmark = custom_benchmark
        self.custom_tasks = custom_tasks
        self.include = include
        self.exclude = exclude
        self.configs = []

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
                name = os.path.basename(config)
                if (name in self.include) or (name.strip(".json") in self.include):
                    continue
                else:
                    exclude.append(config)
            for config in exclude:
                self.configs.remove(config)

        if self.exclude:
            exclude = []
            for config in self.configs:
                name = os.path.basename(config)
                if (name in self.exclude) or (name.strip(".json") in self.exclude):
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
            termination_threshold=self.termination_threshold,
            termination_patience=self.termination_patience,
            termination_exit=False,
            replay_size=self.replay_size,
            replay_purge=self.replay_purge,
        )

    def score(self, *args, **kwargs):
        output = super().score(*args, **kwargs)
        if self.finished:
            # Move on to the next task if available
            if self.configs:
                logger.info("Moving on to next task ...")
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
