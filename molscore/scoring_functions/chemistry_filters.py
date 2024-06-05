import logging
import os
from typing import Dict, List, Union

from molscore.scoring_functions.utils import read_smiles
from moleval.metrics.chemistry_filters import ChemistryFilter as CF

logger = logging.getLogger("chemistry_filter")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class ChemistryFilter:
    """Basic and Target filters (B&T-CF) returns 1 is passed, or 0 if failed."""

    return_metrics = ["CF"]

    def __init__(
        self,
        prefix: str = "B&T",
        ref_smiles: Union[os.PathLike, str, list] = None,
        n_jobs = 1,
        **kwargs,
    ):
        """
        :param prefix: Description
        """
        self.prefix = prefix.strip().replace(" ", "_")
        # If file path provided, load smiles.
        if isinstance(ref_smiles, str):
            self.ref_smiles = read_smiles(ref_smiles)
        else:
            assert isinstance(ref_smiles, list) and (
                len(ref_smiles) > 0
            ), "None list or empty list provided"
            self.ref_smiles = ref_smiles
        # Preprocess filters
        logger.info("Pre-processing Chemistry Filters, this may take a few minutes")
        self.filter = CF(target=self.ref_smiles, n_jobs=n_jobs)

    def __call__(self, smiles: list, **kwargs) -> List[Dict]:
        """
        :param smiles: List of SMILES
        :return: List of dictionaries with return metrics
        """
        results = []
        for smi in smiles:
            passes_filter = self.filter.filter_molecule(
                mol=smi, basic=True, target=True
            )
            results.append(
                {
                    "smiles": smi,
                    f"{self.prefix}_CF": int(passes_filter),
                }
            )
        return results
