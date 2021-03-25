"""
Adapted from
https://github.com/MolecularAI/Reinvent
"""

from rdkit import Chem
from functools import partial
from multiprocessing import Pool


class SubstructureFilters:
    """
    Score structures to penalise undesirable substructures in a molecule (0 returned for a match)
    """
    return_metrics = ['substruct_filt']

    def __init__(self, prefix: str, az_filters: bool = False, custom_filters: list = [],
                 n_jobs: int = 1, **kwargs):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., PAINS)
        :param az_filters: Run filters specified by AstraZeneca in REINVENT publication
        (https://github.com/MolecularAI/Reinvent)
        :param custom_filters: A list of SMARTS to define custom substructure filters.
        :param n_jobs: Number of python.multiprocessing jobs for multiprocessing
        :param kwargs:
        """
        self.prefix = prefix.replace(" ", "_")
        self.n_jobs = n_jobs
        self.smarts = []
        az_smarts = [
            "[*;r8]",
            "[*;r9]",
            "[*;r10]",
            "[*;r11]",
            "[*;r12]",
            "[*;r13]",
            "[*;r14]",
            "[*;r15]",
            "[*;r16]",
            "[*;r17]",
            "[#8][#8]",
            "[#6;+]", "[#7;!n][S;!$(S(=O)=O)]",
            "[#7;!n][#7;!n]",
            "C#C",
            "C(=[O,S])[O,S]",
            "[#7;!n][C;!$(C(=[O,N])[N,O])][#16;!s]",
            "[#7;!n][C;!$(C(=[O,N])[N,O])][#7;!n]",
            "[#7;!n][C;!$(C(=[O,N])[N,O])][#8;!o]",
            "[#8;!o][C;!$(C(=[O,N])[N,O])][#16;!s]",
            "[#8;!o][C;!$(C(=[O,N])[N,O])][#8;!o]",
            "[#16;!s][C;!$(C(=[O,N])[N,O])][#16;!s]",
            "[#16][#16]",
            "[#7;!n][S;!$(S(=O)=O)]",
            "[#7;!n][#7;!n]",
            "C#C",
            "C(=[O,S])[O,S]",
            "[#7;!n][C;!$(C(=[O,N])[N,O])][#16;!s]",
            "[#7;!n][C;!$(C(=[O,N])[N,O])][#7;!n]",
            "[#7;!n][C;!$(C(=[O,N])[N,O])][#8;!o]",
            "[#8;!o][C;!$(C(=[O,N])[N,O])][#16;!s]",
            "[#8;!o][C;!$(C(=[O,N])[N,O])][#8;!o]",
            "[#16;!s][C;!$(C(=[O,N])[N,O])][#16;!s]"
        ]
        if az_filters:
            self.smarts += az_smarts
        if len(custom_filters) > 0:
            self.smarts += custom_filters

    @staticmethod
    def match_substructure(smi: str, smarts_filters: list):
        """
        Method to return a score for a given SMILES string and SMARTS patterns as filters
         (static method for easier multiprocessing)
        :param smi: SMILES string
        :param smarts_filters: List of SMARTS strings
        :return: (SMILES, score)
        """
        mol = Chem.MolFromSmiles(smi)
        if mol:
            match = any([mol.HasSubstructMatch(Chem.MolFromSmarts(sub)) for sub in
                         smarts_filters if Chem.MolFromSmarts(sub)])
            match = not match  # revert, not matching substructure filters should be good i.e. 1
        else:
            match = 0
        return smi, int(match)

    def __call__(self, smiles: list, **kwargs):
        """
        Calculate scores for SubstructureFilters given a list of SMILES.
        :param smiles: A list of SMILES strings
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        with Pool(self.n_jobs) as pool:
            match_substructure_p = partial(self.match_substructure, smarts_filters=self.smarts)
            results = [{'smiles': smi, f'{self.prefix}_substruct_filt': match}
                       for smi, match in pool.imap(match_substructure_p, smiles)]
        return results
