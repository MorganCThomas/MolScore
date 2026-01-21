"""
Interface with AiZynthfinder https://github.com/MolecularAI/aizynthfinder
"""

import logging
import os
from pathlib import Path
from typing import Optional

from molscore import resources
from molscore.scoring_functions.base import BaseServerSF
from molscore.utils.consts import CACHE_DIR

logger = logging.getLogger("aizynthfinder")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class AiZynthFinder(BaseServerSF):
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
        env_engine: str = "mamba",
        iteration_limit: int = 100,
        time_limit: int = 120,
        return_first: bool = True,
        max_transforms: int = 6,
        exclude_target_from_stock: bool = False,
        use_rdchiral: bool = True,
        cutoff_cumulative: float = 0.995,
        cutoff_number: int = 50,
        C: float = 1.4,
        custom_smiles: Optional[os.PathLike] = None,
    ):
        """
        :param prefix: Prefix to identify scoring function instance if used multiple times
        :param env_engine: Environment engine [conda, mamba]
        :param iteration_limit: The maximum number of iterations for the tree search
        :param return_first: If true, the tree search will be terminated as soon as one solution is found
        :param time_limit: The maximum number of seconds to complete the tree search
        :param C: The C value used to balance exploitation and exploration in the upper confidence bound score of the nodes
        :param cutoff_cumulative: The accumulative probability of the suggested templates is capped at this value, all other templates above this threshold are discarded
        :param cutoff_number: The maximum number of templates that will be returned from the expansion policy
        :param max_transforms: The maximum depth of the search tree
        :param exclude_target_from_stock: If the target is in stock it will be broken down if this property is True
        :param use_rdchiral: If true, will apply templates with RDChiral, otherwise RDKit will be used
        :param custom_smiles: A .smi file with custom SMILES strings to add to the stock building blocks
        """
        server_kwargs = {
            "iteration_limit": iteration_limit,
            "time_limit": time_limit,
            "max_transforms": max_transforms,
            "cutoff_cumulative": cutoff_cumulative,
            "cutoff_number": cutoff_number,
            "C": C,
        }

        # Add boolean args which will be passed as flags
        if return_first:
            server_kwargs["return_first"] = ""
        if exclude_target_from_stock:
            server_kwargs["exclude_target_from_stock"] = ""
        if use_rdchiral:
            server_kwargs["use_rdchiral"] = ""
        if custom_smiles is not None:
            server_kwargs["custom_smiles"] = str(custom_smiles)

        # Defaults
        env_name = "ms_aizynth"
        env_path = resources.files("molscore.data.models.aizynth").joinpath(
            "environment.yml"
        )
        server_path = resources.files("molscore.scoring_functions.servers").joinpath(
            "aizynth_server.py"
        )
        grace_period = 120

        # Check if .pidgin_data exists
        LOCAL_CACHE_DIR = CACHE_DIR / "aizynth"
        if not LOCAL_CACHE_DIR.exists():
            logger.warning(
                f"""
{LOCAL_CACHE_DIR} not found, AiZynthFinder data needs to be downloaded.
If no environment exists, please run:
{env_engine} env create -f {env_path}
Then run:
{env_engine} activate {env_name}
python {server_path}
"""
            )

        super().__init__(
            prefix=prefix,
            env_engine="mamba",
            env_name=env_name,
            env_path=env_path,
            server_path=server_path,
            server_kwargs=server_kwargs,
            server_grace=grace_period,
        )


if __name__ == "__main__":
    # Example usage
    smiles = ["CCO", "CCN(CC)CC", "C1=CC=CC=C1"]
    directory = Path("/tmp/aizynth_test")
    directory.mkdir(parents=True, exist_ok=True)
    aizynth = AiZynthFinder()
    results = aizynth(smiles=smiles, directory=directory)
    for res in results:
        print(res)
