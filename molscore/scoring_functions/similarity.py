import argparse
import logging
import os
from functools import partial
from typing import Union

import numpy as np

from molscore.scoring_functions.utils import (
    Fingerprints,
    Pool,
    SimilarityMeasures,
    get_mol,
)

logger = logging.getLogger("tanimoto")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class MolecularSimilarity:
    """
    Score structures based on Similarity to reference structures
    """

    return_metrics = ["Sim"]

    def __init__(
        self,
        prefix: str,
        ref_smiles: Union[list, str, os.PathLike],
        fp: str = "ECFP4",
        bits: int = 1024,
        similarity_measure: str = "Tanimoto",
        thresh: float = None,
        method: str = "mean",
        n_jobs: int = 1,
        **kwargs,
    ):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param ref_smiles: List of SMILES or path to SMILES file with no header (.smi)
        :param fp: Type of fingerprint used to featurize the molecule [ECFP4, ECFP4c, FCFP4, FCFP4c, ECFP6, ECFP6c, FCFP6, FCFP6c, Avalon, MACCSkeys, AP, hashAP, hashTT, RDK5, RDK6, RDK7, PHCO]
        :param bits: Length of fingerprints (if relevant)
        :param similarity_measure: Type of similarity Measure [AllBit, Asymmetric, BraunBlanquet, Cosine, McConnaughey, Dice, Kulczynski, Russel, OnBit, RogotGoldberg, Sokal, Tanimoto]
        :param thresh: If provided check if similarity is above threshold, binarising the similarity coefficients
        :param method: 'mean' or 'max' ('max' is equiv. singler nearest neighbour) [mean, max]
        :param n_jobs: Number of python.multiprocessing jobs for multiprocessing
        :param kwargs:
        """
        self.prefix = prefix.replace(" ", "_")
        assert method in ["max", "mean"]
        self.fp = fp
        self.similarity_measure = similarity_measure
        self.thresh = thresh
        self.method = method
        self.nBits = bits
        self.n_jobs = n_jobs

        # If file path provided, load smiles.
        if isinstance(ref_smiles, str):
            with open(ref_smiles, "r") as f:
                self.ref_smiles = f.read().splitlines()
        else:
            assert isinstance(ref_smiles, list) and (
                len(ref_smiles) > 0
            ), "None list or empty list provided"
            self.ref_smiles = ref_smiles

        # Convert ref smiles to mols
        self.ref_mols = [
            get_mol(smi) for smi in self.ref_smiles if get_mol(smi) is not None
        ]
        if len(self.ref_smiles) != len(self.ref_mols):
            logger.warning(
                f"{len(self.ref_mols)}/{len(ref_smiles)} query smiles converted to mol successfully"
            )

        # Convert ref mols to ref fps
        self.ref_fps = [
            Fingerprints.get(mol, self.fp, self.nBits, asarray=False)
            for mol in self.ref_mols
        ]

    @staticmethod
    def calculate_sim(
        smi: str,
        ref_fps: np.ndarray,
        fp: int,
        nBits: int,
        similarity_measure: str,
        thresh: float,
        method: str,
    ):
        """
        Calculate the Tanimoto coefficient given a SMILES string and list of
         reference fps
        :param smi: SMILES string
        :param ref_fps: ndarray of reference bit vectors
        :param fp: Type of fingerprint used to featurize the molecule
        :param nBits: Number of Morgan fingerprint bits
        :param similarity_measure: Type of measurement used to calculate similarity
        :param thresh: If provided check if similarity is above threshold binarising the similarity coefficients
        :param method: 'mean' or 'max'
        :return: (SMILES, Tanimoto coefficient)
        """
        similarity_measure = SimilarityMeasures.get(similarity_measure, bulk=True)

        mol = get_mol(smi)
        if mol is not None:
            fp = Fingerprints.get(mol, fp, nBits, asarray=False)
            sim_vec = similarity_measure(fp, ref_fps)
            if thresh:
                sim_vec = [sim >= thresh for sim in sim_vec]

            if method == "mean":
                sim = float(np.mean(sim_vec))
            elif method == "max":
                sim = float(np.max(sim_vec))
            else:
                sim = 0.0
        else:
            sim = 0.0

        return smi, sim

    def __call__(self, smiles: list, **kwargs):
        """
        Calculate scores for Tanimoto given a list of SMILES.
        :param smiles: List of SMILES strings
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        with Pool(self.n_jobs) as pool:
            calculate_sim_p = partial(
                self.calculate_sim,
                ref_fps=self.ref_fps,
                fp=self.fp,
                nBits=self.nBits,
                thresh=self.thresh,
                similarity_measure=self.similarity_measure,
                method=self.method,
            )
            results = [
                {"smiles": smi, f"{self.prefix}_Sim": sim}
                for smi, sim in pool.imap(calculate_sim_p, smiles)
            ]
        return results


# Adding for backwards compatability
class TanimotoSimilarity(MolecularSimilarity):
    """
    Score structures based on Tanimoto similarity to reference structures
    """

    def __init__(
        self,
        prefix: str,
        ref_smiles: Union[list, os.PathLike],
        fp: str = "ECFP4",
        bits: int = 1024,
        method: str = "mean",
        n_jobs: int = 1,
        **kwargs,
    ):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param ref_smiles: List of SMILES or path to SMILES file with no header (.smi)
        :param fp: Type of fingerprint used to featurize the molecule [ECFP4, ECFP4c, FCFP4, FCFP4c, ECFP6, ECFP6c, FCFP6, FCFP6c, Avalon, MACCSkeys, AP, hashAP, hashTT, RDK5, RDK6, RDK7, PHCO]
        :param bits: Length of fingerprints (if relevant)
        :param similarity_measure: Type of similarity Measure [AllBit, Asymmetric, BraunBlanquet, Cosine, McConnaughey, Dice, Kulczynski, Russel, OnBit, RogotGoldberg, Sokal, Tanimoto]
        :param method: 'mean' or 'max' ('max' is equiv. singler nearest neighbour) [mean, max]
        :param n_jobs: Number of python.multiprocessing jobs for multiprocessing
        :param kwargs:
        """
        # Backwards compatability for previous parameters of radius, count and features
        if all(p in kwargs.keys() for p in ["radius", "counts", "features"]):
            if kwargs["features"]:
                fp = "F" + "CFP" + str(kwargs["radius"] * 2)
            else:
                fp = "E" + "CFP" + str(kwargs["radius"] * 2)

            if kwargs["counts"]:
                fp += "c"

        super().__init__(
            prefix=prefix,
            ref_smiles=ref_smiles,
            fp=fp,
            bits=bits,
            method=method,
            n_jobs=n_jobs,
            **kwargs,
        )


if __name__ == "__main__":
    # Read in CLI arguments
    parser = argparse.ArgumentParser(
        description="Calculate the Maximum or Average Tanimoto similarity for a set of "
        "molecules to a set of reference molecules",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(title="Running mode", dest="mode")
    run = subparsers.add_parser(
        "run", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    run.add_argument(
        "--prefix", type=str, default="test", help="Prefix to returned metrics"
    )
    run.add_argument("--ref_smiles", help="Path to smiles file (.smi) for ref smiles")
    run.add_argument(
        "--query_smiles", help="Path to smiles file (.smi) for query smiles"
    )
    run.add_argument(
        "--fp",
        default="ECFP4",
        help="What fingerprint to use",
        choices=[
            "ECFP4",
            "ECFP4c",
            "FCFP4",
            "FCFP4c",
            "ECFP6",
            "ECFP6c",
            "FCFP6",
            "FCFP6c",
            "Avalon",
            "MACCSkeys",
            "hashAP",
            "hashTT",
            "RDK5",
            "RDK6",
            "RDK7",
        ],
    )
    run.add_argument("--bits", default=1024, help="Morgan fingerprint bit size")
    run.add_argument(
        "--similarity_measure",
        default="Tanimoto",
        help="What similarity measure to use",
        choices=[
            "AllBit",
            "Asymmetric",
            "BraunBlanquet",
            "Cosine",
            "McConnaughey",
            "Dice",
            "Kulczynski",
            "Russel",
            "OnBit",
            "RogotGoldberg",
            "Sokal",
            "Tanimoto",
        ],
    )
    run.add_argument(
        "--method",
        type=str,
        choices=["mean", "max"],
        default="mean",
        help="Use mean or max similarity to ref smiles",
    )
    run.add_argument("--n_jobs", default=1, type=int, help="How many cores to use")
    test = subparsers.add_parser("test")
    args = parser.parse_args()

    # Run mode
    if args.mode == "run":
        from molscore.tests import MockGenerator

        mg = MockGenerator(seed_no=123)

        if args.ref_smiles is None:
            args.ref_smiles = mg.sample(10)

        ts = TanimotoSimilarity(
            prefix=args.prefix,
            ref_smiles=args.ref_smiles,
            fp=args.fp,
            bits=args.bits,
            similarity_measure=args.similarity_measure,
            method=args.method,
            n_jobs=args.n_jobs,
        )

        if args.query_smiles is None:
            _ = [print(o) for o in ts(mg.sample(5))]
        else:
            with open(args.query_smiles, "r") as f:
                qsmiles = f.read().splitlines()
            _ = [print(o) for o in ts(qsmiles)]

    # Test mode
    elif args.mode == "test":
        import sys
        import unittest

        from molscore.tests import test_tanimoto

        # Remove CLI arguments
        for i in range(len(sys.argv) - 1):
            sys.argv.pop()
        # Get and run tests
        suite = unittest.TestLoader().loadTestsFromModule(test_tanimoto)
        unittest.TextTestRunner().run(suite)

    # Print usage
    else:
        print("Please specify running mode: 'run' or 'test'")
