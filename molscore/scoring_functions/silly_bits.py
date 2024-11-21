import logging
import os
from collections import defaultdict
from functools import partial
from typing import Union

from rdkit.Chem import AllChem as Chem

from molscore.scoring_functions.utils import Pool, get_mol, read_smiles

logger = logging.getLogger("silly_bits")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class SillyBits:
    """Ratio of fingerprint bits not found in a reference dataset based on https://github.com/PatWalters/silly_walks"""

    return_metrics = ["silly_ratio"]

    def __init__(
        self,
        prefix: str,
        reference_smiles: os.PathLike,
        radius: int = 2,
        n_jobs=1,
        **kwargs,
    ):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., ChEMBLbits)
        :param reference_mols: List of SMILES or RDKit Mols
        :param radius: Morgan fingerprint radius
        :param n_jobs: Number of jobs for multiprocessing
        """
        self.prefix = prefix.replace(" ", "_")
        self.count_dict = defaultdict(int)
        self.radius = radius
        self.n_jobs = n_jobs
        self.mapper = Pool(self.n_jobs, return_map=True)
        # Load reference dataset
        reference_mols = read_smiles(reference_smiles)
        # Convert to mols from reference dataset and count fp bits
        logger.info("Pre-processing SillyBits reference dataset")
        bit_counts = [bits for bits in self.mapper(self.count_bits, reference_mols)]
        for count_dict in bit_counts:
            for k, v in count_dict.items():
                self.count_dict[k] += v

    @staticmethod
    def count_bits(mol):
        count_dict = {}
        mol = get_mol(mol)
        if mol is not None:
            fp = Chem.GetMorganFingerprint(mol, 2)
            for k, v in fp.GetNonzeroElements().items():
                count_dict[k] = v
        return count_dict

    @staticmethod
    def _score(mol: Union[str, Chem.rdchem.Mol], count_dict):
        mol = get_mol(mol)
        if mol is not None:
            bi = {}
            fp = Chem.GetMorganFingerprint(mol, 2, bitInfo=bi)
            on_bits = fp.GetNonzeroElements().keys()
            silly_bits = [bit for bit in on_bits if count_dict[bit] == 0]
            score = len(silly_bits) / len(on_bits)
            return score, silly_bits, bi

    def __call__(self, smiles, **kwargs):
        results = []
        pfunc = partial(self._score, count_dict=self.count_dict)
        scores = [s for s in self.mapper(pfunc, smiles)]
        for smi, score in zip(smiles, scores):
            if score is not None:
                results.append({"smiles": smi, f"{self.prefix}_silly_ratio": score[0]})
            else:
                # Going to provide one because invalid molecules are very silly
                results.append({"smiles": smi, f"{self.prefix}_silly_ratio": 1.0})
        return results
