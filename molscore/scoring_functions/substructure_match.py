"""
Adapted from
https://github.com/MolecularAI/Reinvent
"""

from rdkit import Chem
from functools import partial
from multiprocessing import Pool


class SubstructureMatch:
    """
    Scoring function class to reward desirable substructures in a molecule.
    """
    def __init__(self, prefix: str, smarts: list = [],
                 n_jobs: int = 1, method: str = 'any', **kwargs):
        """
        Scoring function class to reward desirable substructures in a molecule.
        :param prefix: Name (to help keep track metrics, if using a scoring function class more than once)
        :param smarts: List of SMARTS strings that define desirable substructures
        :param n_jobs: Number of jobs for multiprocessing
        :param method: To give reward for 'any' match, or only for 'all' matches
        :param kwargs: Ignored
        """
        self.prefix = prefix.replace(" ", "_")
        self.n_jobs = n_jobs
        self.smarts = smarts
        assert method in ['any', 'all']
        self.method = method

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
            if method == 'any':
                match = any([mol.HasSubstructMatch(Chem.MolFromSmarts(sub)) for sub in
                             smarts if Chem.MolFromSmarts(sub)])
            if method == 'all':
                match = all([mol.HasSubstructMatch(Chem.MolFromSmarts(sub)) for sub in
                             smarts if Chem.MolFromSmarts(sub)])
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
        with Pool(self.n_jobs) as pool:
            match_substructure_p = partial(self.match_substructure, smarts=self.smarts, method=self.method)
            results = [{'smiles': smi, f'{self.prefix}_substructure_match': match}
                       for smi, match in pool.imap(match_substructure_p, smiles)]
        return results
