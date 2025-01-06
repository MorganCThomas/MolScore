"""
Adapted from
https://github.com/MolecularAI/Reinvent
"""

import os
from functools import partial
from typing import Union, Optional

from rdkit import Chem

from molscore.scoring_functions.utils import Pool


class SubstructureMatch:
    """
    Score structures based on desirable substructures in a molecule (1 returned for a match)
    """

    return_metrics = ["substruct_match"]

    def __init__(
        self,
        prefix: str,
        smarts: Union[list, os.PathLike],
        n_jobs: Optional[int] = 1,
        method: str = "any",
        **kwargs,
    ):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., Benzimidazole)
        :param smarts: List of SMARTS or path to SMARTS file (format of a .smi i.e., txt with one row, no header)
        :param n_jobs: Number of python.multiprocessing jobs for multiprocessing
        :param method: 'any' or 'all': Give reward for 'any' match, or only for 'all' matches (reward is 1 or 0)
        :param kwargs:
        """
        self.prefix = prefix.replace(" ", "_")
        self.n_jobs = n_jobs
        self.mapper = Pool(self.n_jobs, return_map=True)
        self.smarts = smarts
        assert method in ["any", "all"]
        self.method = method

        # If file path provided, load smiles.
        if isinstance(smarts, str):
            with open(smarts, "r") as f:
                self.smarts = f.read().splitlines()
        else:
            assert isinstance(smarts, list) and (
                len(smarts) > 0
            ), "None list or empty list provided"
            self.smarts = smarts

    @staticmethod
    def match_substructure(smi: str, smarts: list, method: str):
        """
        Method to return a score for a given SMILES string, SMARTS patterns and method ('all' or 'any')
         (static method for easier multiprocessing)
        :param smi: SMILES string
        :param smarts: List of SMARTS strings
        :param method: Require to match either 'any' or 'all' SMARTS
        :return: (SMILES, score)
        """
        mol = Chem.MolFromSmiles(smi)
        if mol:
            if method == "any":
                match = any(
                    [
                        mol.HasSubstructMatch(Chem.MolFromSmarts(sub))
                        for sub in smarts
                        if Chem.MolFromSmarts(sub)
                    ]
                )
            if method == "all":
                match = all(
                    [
                        mol.HasSubstructMatch(Chem.MolFromSmarts(sub))
                        for sub in smarts
                        if Chem.MolFromSmarts(sub)
                    ]
                )
        else:
            match = 0
        return smi, int(match)

    def __call__(self, smiles: list, **kwargs):
        """
        Calculate scores for SubstructureMatch given a list of SMILES.
        :param smiles: List of SMILES strings
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        match_substructure_p = partial(
                self.match_substructure, smarts=self.smarts, method=self.method
            )
        results = [
                {"smiles": smi, f"{self.prefix}_substruct_match": match}
                for smi, match in self.mapper(match_substructure_p, smiles)
            ]
        return results
