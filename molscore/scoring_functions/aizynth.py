"""
Interface with AiZynthfinder https://github.com/MolecularAI/aizynthfinder
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Final

import pandas as pd
from aizynthfinder.aizynthfinder import AiZynthFinder as AIZynth
from rdkit import Chem

from molscore.scoring_functions.utils import timedSubprocess
from molscore.utils.consts import CACHE_DIR

logger = logging.getLogger("aizynthfinder")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


LOCAL_CACHE_DIR: Final = CACHE_DIR / "aizynth"
LOCAL_CACHE_DIR.mkdir(exist_ok=True, parents=True)


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
        iteration_limit: int = 100,
        time_limit: int = 120,
        return_first: bool = True,
        max_transforms: int = 6,
        exclude_target_from_stock: bool = False,
        use_rdchiral: bool = True,
        cutoff_cumulative: float = 0.995,
        cutoff_number: int = 50,
        C: float = 1.4,
        custom_smiles: list[str] = [],
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
        """
        self.prefix = prefix.replace(" ", "_")
        self.config_file = tempfile.NamedTemporaryFile(mode="w+t", suffix="_config.yml")
        self.subprocess = timedSubprocess(timeout=None, shell=False)
        self.iteration_limit = iteration_limit
        self.time_limit = time_limit
        self.return_first = return_first
        self.max_transforms = max_transforms
        self.exclude_target_from_stock = exclude_target_from_stock
        self.use_rdchiral = use_rdchiral
        self.cutoff_cumulative = cutoff_cumulative
        self.cutoff_number = cutoff_number
        self.C = C

        self._download_data_if_not_exists()

        self.custom_inchikeys = None
        if len(custom_smiles) > 0:
            self.custom_inchikeys = self.smiles_to_inchikeys(custom_smiles)

        self.finder = self._setup_finder()

    def smiles_to_inchikeys(self, smiles: list[str]) -> list[str]:
        inchikeys = []
        for smi in smiles:
            rdmol = Chem.MolFromSmiles(smi)
            if rdmol is None:
                raise ValueError(f"Invalid SMILES: {smi}")
            inchikey = Chem.MolToInchiKey(rdmol)
            inchikeys.append(inchikey)
        return inchikeys

    def add_custom_stock(
        self, custom_inchikeys: list[str], config_dict: dict[str, Any]
    ) -> dict[str, Any]:
        custom_stock_file = str(LOCAL_CACHE_DIR / "custom_stock.hdf5")
        custom_stock_df = pd.DataFrame(custom_inchikeys, columns=["inchi_key"])
        custom_stock_df.to_hdf(custom_stock_file, key="table")
        config_dict["stock"]["files"]["custom"] = custom_stock_file
        return config_dict

    def _setup_finder(self) -> AIZynth:
        """
        Set up the AiZynthFinder instance.
        """
        config_dict = self.get_config_dict()
        # Add any custom molecules as stock building blocks
        if self.custom_inchikeys:
            config_dict = self.add_custom_stock(self.custom_inchikeys, config_dict)
        finder = AIZynth(configdict=config_dict)
        finder.expansion_policy.select_all()
        finder.filter_policy.select_all()
        finder.config.stock.select_all()
        return finder

    def _download_data_if_not_exists(self):
        """
        Download data for AIZynthFinder if it does not exist locally.
        """
        if (LOCAL_CACHE_DIR / "zinc_stock.hdf5").exists():
            return

        logger.info("Local data not found for AIZynthFinder. Downloading")
        os.system(f"download_public_data {LOCAL_CACHE_DIR!s}")

    def get_config_dict(self) -> dict[str, Any]:
        """
        Returns:
            Configuration dictionary.
        """
        config_dict = {
            "search": {
                "algorithm": "mcts",
                "algorithm_config": {"C": self.C},
                "max_transforms": self.max_transforms,
                "iteration_limit": self.iteration_limit,
                "return_first": self.return_first,
                "time_limit": self.time_limit,
                "exclude_target_from_stock": self.exclude_target_from_stock,
            },
            "expansion": {
                "uspto": {
                    "model": str(LOCAL_CACHE_DIR / "uspto_model.onnx"),
                    "template": str(LOCAL_CACHE_DIR / "uspto_templates.csv.gz"),
                    "use_rdchiral": self.use_rdchiral,
                    "cutoff_cumulative": self.cutoff_cumulative,
                    "cutoff_number": self.cutoff_number,
                },
                "ringbreaker": {
                    "model": str(LOCAL_CACHE_DIR / "uspto_ringbreaker_model.onnx"),
                    "template": str(
                        LOCAL_CACHE_DIR / "uspto_ringbreaker_templates.csv.gz"
                    ),
                    "use_rdchiral": self.use_rdchiral,
                    "cutoff_cumulative": self.cutoff_cumulative,
                    "cutoff_number": self.cutoff_number,
                },
            },
            "stock": {"zinc": str(LOCAL_CACHE_DIR / "zinc_stock.hdf5")},
            "filter": {"uspto": str(LOCAL_CACHE_DIR / "uspto_filter_model.onnx")},
        }
        return config_dict

    def run(
        self, query_smi: str, show_progress: bool = False
    ) -> dict[str, int | float]:
        """
        Run the AiZynthFinder on the query SMILES and return statistics.

        Args:
            query_smi: SMILES string to query.
            show_progress: Show progress of the tree search.

        Returns:
            Statistics dictionary.
        """
        self.finder.target_smiles = query_smi

        self.finder.tree_search(show_progress=show_progress)
        self.finder.build_routes()

        stats = self.finder.extract_statistics()

        stats["precursors_in_stock_ratio"] = (
            stats["number_of_precursors_in_stock"] / stats["number_of_precursors"]
        )
        return stats

    def __call__(self, smiles: list[str], **kwargs) -> dict[str, Any]:
        """
        Calculate AiZynthfinder for a list of smiles.
        :param smiles: List of SMILES strings.
        :return: Dictionary with results for each SMILES.
        """
        results = []
        for smi in smiles:
            result = self.score(smi, **kwargs)
            results.append(result)
        return results

    def score(self, smi: str, **kwargs) -> dict[str, Any]:
        """
        Generate a predicted retrosynthesis route for a given SMILES string, returning
        the number of steps, whether the route is solved, and score the route based on no. of
        precursors in stock + the length of the route (native to aizynthfinder). Also returns
        the best predicted route as a JSON string and an image of the route.

        Args:
            smi: SMILES string

        Returns:
            The result dictionary.
        """

        results = self.run(smi)
        # If the synthetic route could be fully planned, the first route is the most complete.
        # If the route is fully planned, the first route is the shortest.
        # This route scoring can be changed in AIZynthFinder. Here we just use the default.
        best_route = self.finder.routes.reaction_trees[0]
        route_json = best_route.to_json()
        # Image of the route can be constructed using
        # from aizynthfinder.utils.image import RouteImageFactory
        # img = RouteImageFactory(yaml.safe_load(r['test_route_json'])).image

        return {
            "smiles": smi,
            f"{self.prefix}_is_solved": results["is_solved"],
            f"{self.prefix}_number_of_steps": results["number_of_steps"],
            f"{self.prefix}_top_score": results["top_score"],
            f"{self.prefix}_route_json": route_json,
        }


if __name__ == "__main__":
    # Example usage
    smiles = ["CCO", "CCN(CC)CC", "C1=CC=CC=C1"]
    directory = Path("/tmp/aizynth_test")
    directory.mkdir(parents=True, exist_ok=True)
    aizynth = AiZynthFinder()
    results = aizynth(smiles=smiles, directory=directory)
    for res in results:
        print(res)
