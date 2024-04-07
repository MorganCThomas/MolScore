"""
Interface with AiZynthfinder https://github.com/MolecularAI/aizynthfinder
"""

import json
import logging
import os
import subprocess
import tempfile
from typing import Union

import yaml

from molscore import resources
from molscore.scoring_functions.utils import check_exe, get_mol, timedSubprocess

logger = logging.getLogger("aizynthfinder")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class AiZynthFinder:
    """
    Computer aided synthesis planning via AiZynthFinder by AstraZeneca
    Genheden et al. https://pubs.rsc.org/en/content/articlelanding/2020/SC/C9SC04944D
    an open-source implementation of
    Segler et al. https://www.nature.com/articles/nature25978
    """

    return_metrics = [
        "is_solved",
        "top_score",
        "number_of_steps",
        "number_of_precursors",
    ]

    def __init__(
        self,
        prefix: str = "AiZynth",
        config_file: Union[str, os.PathLike] = None,
        policy: str = "full_uspto_rollout_policy",
        templates: str = "full_uspto_unique_templates",
        filter_policy: str = "filter_policy_all",
        stock: Union[str, os.PathLike] = "zinc_stock",
        n_jobs: int = 1,
        iteration_limit: int = 100,
        return_first: bool = False,
        time_limit: int = 120,
        C: float = 1.4,
        cutoff_cumulative: float = 0.995,
        cutoff_number: int = 50,
        max_transforms: int = 6,
        exclude_target_from_stock: bool = False,
        use_rdchiral: bool = True,
        env_engine="mamba",
        **kwargs,
    ):
        """
        :param prefix: Prefix to identify scoring function instance if used multiple times
        :param policy: Expansion policy model checkpoint file [full_uspto_rollout_policy]
        :param templates: Expansion policy model template file [full_uspto_unique_templates]
        :param filter_policy: Additional policy filter [filter_policy_all, filter_policy_random, filter_policy_recommender, filter_policy_strict, None]
        :param stock: Stock file listing all available precursors according to a stock [zinc_stock]
        :param n_jobs: Number of cores for aizynthfinder to utilise
        :param iteration_limit: The maximum number of iterations for the tree search
        :param return_first: If true, the tree search will be terminated as soon as one solution is found
        :param time_limit: The maximum number of seconds to complete the tree search
        :param C: The C value used to balance exploitation and exploration in the upper confidence bound score of the nodes
        :param cutoff_cumulative: The accumulative probability of the suggested templates is capped at this value, all other templates above this threshold are discarded
        :param cutoff_number: The maximum number of templates that will be returned from the expansion policy
        :param max_transforms: The maximum depth of the search tree
        :param exclude_target_from_stock: If the target is in stock it will be broken down if this property is True
        :param use_rdchiral: If true, will apply templates with RDChiral, otherwise RDKit will be used
        :param env_engine: Engine to be used to handle python environments
        """
        self.prefix = prefix.replace(" ", "_")
        self.config_file = tempfile.NamedTemporaryFile(mode="w+t", suffix="_config.yml")
        self.n_jobs = n_jobs
        self.subprocess = timedSubprocess(timeout=None, shell=False)
        self.env = "aizynth-env"

        # Set environment engine
        if (env_engine == "mamba") and check_exe("mamba"):
            self.engine = "mamba"
        elif check_exe("conda"):
            self.engine = "conda"
        else:
            raise ValueError(
                "Could not find mamba or conda executables needed to create and run this environment"
            )

        # Check/create AiZynth Environment
        if not self._check_env():
            logger.warning(
                f"Failed to identify {self.env}, attempting to create it automatically (this may take several minutes)"
            )
            self._create_env()
            logger.info(f"{self.env} successfully created")
        else:
            logger.info(f"Found existing {self.env}")

        # Load policy
        def try_resources(input):
            try:
                with resources.path(
                    "molscore.data.models.aizynth", resource=f"{input}.hdf5"
                ) as r:
                    output = str(r)
                    return output
            except FileNotFoundError:
                if os.path.exists(input):
                    return input
                else:
                    raise FileNotFoundError(
                        f"File not found for {input}, please specify."
                    )

        # Write config policy to tempfile
        if (config_file is None) or (config_file in [".", "None"]):
            self.config = {
                "properties": {
                    "iteration_limit": iteration_limit,
                    "return_first": return_first,
                    "time_limit": time_limit,
                    "C": C,
                    "cutoff_cumulative": cutoff_cumulative,
                    "cutoff_number": cutoff_number,
                    "max_transforms": max_transforms,
                    "exclude_target_from_stock": exclude_target_from_stock,
                    "use_rdchiral": use_rdchiral,
                },
                "policy": {
                    "files": {
                        "my_policy": [try_resources(policy), try_resources(templates)]
                    },
                },
                "stock": {"files": {"my_stock": try_resources(stock)}},
            }
            # Add filter policy if specified
            if filter_policy and (filter_policy != "None"):
                self.filter_policy = True
                self.config.update(
                    {"filter": {"files": {"my_policy": try_resources(filter_policy)}}}
                )

            # Write to tempfile
            self.config_file.write(yaml.dump(self.config))
            self.config_file.flush()

        else:
            self.config_file = config_file
            with open(self.config_file, "rt") as f:
                self.config = yaml.load(f, Loader=yaml.SafeLoader)
        # Write to log
        logger.info(
            f"AiZynthFinder instantiated with the following configuration:\n{yaml.dump(self.config)}"
        )

    def _check_env(self):
        cmd = "{self.engine} info --envs"
        out = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
        envs = [line.split(" ")[0] for line in out.stdout.decode().splitlines()[2:]]
        return self.env in envs

    def _create_env(self):
        cmd = (
            f'{self.engine} create "python>=3.8,<3.10" -n {self.env} -y ; '
            f"{self.engine} run -n {self.env} pip install aizynthfinder"
        )
        try:
            out = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(out.stderr.decode())
            logger.error(
                f"Failed to create {self.env} automatically please install as per instructions https://github.com/MolecularAI/aizynthfinder"
            )
            raise e

    def score(self, smiles, directory, **kwargs):
        """
        Calculate AiZynthfinder for a list of smiles.
        :param smiles: List of SMILES strings.
        :param directory: Directory to save files and logs into
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        directory = os.path.abspath(directory)
        # Populate results with 0.0 placeholder
        results = [{"smiles": smi} for smi in smiles]
        for result in results:
            result.update(
                {f"{self.prefix}_{metric}": 0.0 for metric in self.return_metrics}
            )
        # Ensure they're valid otherwise AiZynthFinder will throw an error
        valid = [i for i, smi in enumerate(smiles) if get_mol(smi)]
        if len(valid) == 0:
            return results
        # Write smiles to a tempfile
        smiles_file = tempfile.NamedTemporaryFile(
            mode="w+t", dir=directory, suffix=".smi"
        )
        for i in valid:
            smiles_file.write(smiles[i] + "\n")
        smiles_file.flush()
        # Specify output file
        output_file = os.path.join(directory, "aizynth_out.json")
        # Submit job to aizynthcli (specify filter policy if not None)
        cmd = (
            f"{self.engine} run -n {self.env} "
            f"aizynthcli --smiles {smiles_file.name} --config {self.config_file.name} --output {output_file} --nproc {int(self.n_jobs)}"
        )  # --filter my_policy
        if self.filter_policy:
            cmd += " --filter my_policy"
        self.subprocess.run(cmd=cmd, cwd=directory)
        # Read in ouput
        with open(output_file, "rt") as f:
            output_data = json.load(f)["data"]
        # Process output
        for i, out in zip(valid, output_data):
            results[i].update(
                {
                    f"{self.prefix}_{metric}": out[metric]
                    for metric in self.return_metrics
                }
            )
            # Also add precursors
            results[i].update(
                {
                    f"{self.prefix}_{metric}": out[metric]
                    for metric in ["precursors_in_stock", "precursors_not_in_stock"]
                }
            )
        return results

    def __call__(self, smiles, directory, **kwargs):
        """
        Calculate AiZynthfinder for a list of smiles.
        :param smiles: List of SMILES strings.
        :param directory: Directory to save files and logs into
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        return self.score(smiles=smiles, directory=directory)
