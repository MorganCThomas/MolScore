import logging
import os
import json
from collections import defaultdict
from functools import partial
from typing import Union, Optional

from rdkit.Chem import AllChem as Chem

from molscore import resources
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
    PRESETS = {
        "DiverseHits": resources.files("molscore.data.models.diversehits").joinpath("guacamol_known_bits.json"),
    }

    def __init__(
        self,
        prefix: str,
        preset: Optional[str] = None,
        reference_smiles: Optional[os.PathLike] = None,
        radius: int = 2,
        n_jobs=1,
        **kwargs,
    ):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., ChEMBLbits)
        :param preset: Preset bit list to use [DiverseHits]
        :param reference_mols: List of SMILES or RDKit Mols
        :param radius: Morgan fingerprint radius
        :param n_jobs: Number of jobs for multiprocessing
        """
        self.prefix = prefix.replace(" ", "_")
        self.count_dict = defaultdict(int)
        self.radius = radius
        self.n_jobs = n_jobs
        self.pool = Pool(self.n_jobs)
        
        # Load / calculate bits
        assert preset or reference_smiles, "Either a preset or a reference dataset must be provided"
        if preset is not None:
            logger.info(f"Loading SillyBits preset: {preset}")
            assert preset in self.PRESETS, f"Preset {preset} not found. Available presets: {list(self.PRESETS.keys())}"
            with open(self.PRESETS[preset], "r") as f:
                preset_bits = json.load(f)
                for b in preset_bits:
                    self.count_dict[b] += 1
        if reference_smiles is not None:
            reference_mols = read_smiles(reference_smiles)
            # Convert to mols from reference dataset and count fp bits
            logger.info("Pre-processing SillyBits reference dataset")
            count_bits = partial(self.count_bits, radius=self.radius)
            bit_counts = self.pool.submit(count_bits, reference_mols)
            for count_dict in bit_counts:
                for k, v in count_dict.items():
                    self.count_dict[k] += v
                    
        # Prepare score function
        self.score_mol = partial(self._score, count_dict=self.count_dict, radius=self.radius)

    @staticmethod
    def count_bits(mol, radius):
        count_dict = {}
        mol = get_mol(mol)
        if mol is not None:
            fp = Chem.GetMorganFingerprint(mol, radius)
            for k, v in fp.GetNonzeroElements().items():
                count_dict[k] = v
        return count_dict

    @staticmethod
    def _score(mol: Union[str, Chem.rdchem.Mol], count_dict, radius):
        mol = get_mol(mol)
        if mol is not None:
            bi = {}
            fp = Chem.GetMorganFingerprint(mol, radius, bitInfo=bi)
            on_bits = fp.GetNonzeroElements().keys()
            silly_bits = [bit for bit in on_bits if count_dict[bit] == 0]
            score = len(silly_bits) / len(on_bits)
            return score, silly_bits, bi

    def __call__(self, smiles, **kwargs):
        results = []
        scores = self.pool.submit(self.score_mol, smiles)
        for smi, score in zip(smiles, scores):
            if score is not None:
                results.append({"smiles": smi, f"{self.prefix}_silly_ratio": score[0]})
            else:
                # Going to provide one because invalid molecules are very silly
                results.append({"smiles": smi, f"{self.prefix}_silly_ratio": 1.0})
        return results
