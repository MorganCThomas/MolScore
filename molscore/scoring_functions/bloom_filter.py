import logging
import os
from typing import Union

import molbloom
import numpy as np

from molscore.scoring_functions.utils import Pool, canonize_smiles, read_smiles

logger = logging.getLogger("bloom_filter")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class BloomFilter:
    """Using MolBloom to estimate presence in a database"""

    return_metrics = [
        "inside",
        "outside",
    ]  # Name of metrics returned so that they can be selected in the config GUI

    def __init__(
        self,
        prefix: str,
        preset: str = None,
        bloom_path: Union[str, os.PathLike] = None,
        smiles_path: Union[str, os.PathLike] = None,
        canonize=True,
        fpr: float = 0.01,
        n_jobs: int = 1,
        **kwargs,
    ):
        """
        :param prefix: Prefix to seperate multiple uses of the same class
        :param preset: Download a preset from MolBloom [zinc20, zinc-instock, zinc-instock-mini, surechembl]
        :param bloom_path: Path to saved the MolBloom database file (with .bloom extension)
        :param smiles_path: Path to a SMILES file to create a new bloom filter
        :param canonize: Canonicalize SMILES if creating molbloom database
        :param fpr: False positive rate for MolBloom if creating molbloom database
        """
        self.prefix = prefix.strip().replace(" ", "_")
        self.preset = preset
        self.smiles_path = os.path.abspath(smiles_path) if smiles_path else None
        self.bloom_path = os.path.abspath(bloom_path) if bloom_path else None
        self.canonize = canonize
        self.fpr = fpr
        self.n_jobs = n_jobs
        self.mapper = Pool(self.n_jobs, return_map=True)
        self.filter = None

        parameters_provided = sum(
            [p is not None for p in [self.preset, self.smiles_path, self.bloom_path]]
        )
        if parameters_provided > 1:
            logger.warning(
                "Multiple parameters provided, precedent for use is preset first, then bloom_path, then smiles_path"
            )

        if self.preset:
            # Download if not downloaded
            assert (
                self.preset in molbloom.catalogs()
            ), f"Preset {self.preset} not found in MolBloom catalog"
            molbloom._load_filter(self.preset)
            self.filter = molbloom._filters[self.preset]

        elif self.bloom_path:
            assert os.path.exists(self.bloom_path), "Bloom path does not exist"
            self.filter = molbloom.BloomFilter(os.path.abspath(self.bloom_path))

        elif self.smiles_path:
            # Load
            smiles_list = read_smiles(os.path.abspath(self.smiles_path))
            # Canonize
            if self.canonize:
                smiles_list = [s for s in self.mapper(canonize_smiles, smiles_list)]
            # Set up bloom filter parameters based on False Positive Rate
            M_bits = -(len(smiles_list) * np.log(self.fpr)) / (np.log(2) ** 2)
            self.filter = molbloom.CustomFilter(M_bits, len(smiles_list), self.prefix)
            # Add to filter or create filter
            for smi in smiles_list:
                self.filter.add(smi)
            # Save filter
            logger.info(f"Saving filter to {self.smiles_path.rsplit('.', 1)[0]}.bloom")
            self.filter.save(self.smiles_path.rsplit(".", 1)[0] + ".bloom")

        else:
            raise ValueError(
                "At least one of preset, smiles_path or bloom_path must be provided"
            )

    def __call__(self, smiles: list, **kwargs):
        results = []
        for smi in smiles:
            inside = int(smi in self.filter)
            results.append(
                {
                    "smiles": smi,
                    f"{self.prefix}_inside": inside,
                    f"{self.prefix}_outside": 1 - inside,
                }
            )
        return results
