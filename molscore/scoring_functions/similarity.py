import logging
import os
import atexit
from functools import partial
from typing import Union

import numpy as np
from Levenshtein import distance as levenshtein

from molscore.scoring_functions.utils import (
    Fingerprints,
    Pool,
    SimilarityMeasures,
    canonize_smiles,
    get_mol,
    timedFunc2,
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
        timeout: int = 60,
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
        :param timeout: Timeout for the scoring to cease and return a score of 0.0
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
        self.timeout = timeout
        self.mapper = Pool(n_jobs, return_map=True)

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
        prefix: str,
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
            sim_vec = []

        result = {"smiles": smi, f"{prefix}_Sim": sim}
        result.update({f"{prefix}_Cmpd{i+1}_Sim": sim for i, sim in enumerate(sim_vec)})

        return result

    def _score(self, smiles: list, **kwargs):
        """
        Calculate scores for Tanimoto given a list of SMILES.
        :param smiles: List of SMILES strings
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        calculate_sim_p = partial(
                self.calculate_sim,
                ref_fps=self.ref_fps,
                fp=self.fp,
                nBits=self.nBits,
                thresh=self.thresh,
                similarity_measure=self.similarity_measure,
                method=self.method,
                prefix=self.prefix,
            )
        
        results = [result for result in self.mapper(calculate_sim_p, smiles)]
        return results

    def __call__(self, smiles: list, **kwargs):
        """
        Calculate scores for Tanimoto given a list of SMILES.
        :param smiles: List of SMILES strings
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        tfunc = timedFunc2(self._score, timeout=self.timeout)
        results = tfunc(smiles)
        if results is None:
            logger.warning(
                f"Timeout of {self.timeout} reached for scoring, returning 0.0"
            )
            results = [{"smiles": smi, f"{self.prefix}_Sim": 0.0} for smi in smiles]

        return results


class LevenshteinSimilarity(MolecularSimilarity):
    """
    Score structures based on the normalized Levenshtein similarity of provided SMILES string(s) to reference structure
    """

    return_metrics = ["Sim"]

    def __init__(
        self,
        prefix: str,
        ref_smiles: Union[list, str, os.PathLike],
        thresh: float = None,
        method: str = "mean",
        n_jobs: int = 1,
        timeout: int = 60,
        **kwargs,
    ):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param ref_smiles: List of SMILES or path to SMILES file with no header (.smi)
        :param thresh: If provided check if similarity is above threshold, binarising the similarity coefficients
        :param method: 'mean' or 'max' ('max' is equiv. singler nearest neighbour) [mean, max]
        :param n_jobs: Number of python.multiprocessing jobs for multiprocessing
        :param timeout: Timeout for the scoring to cease and return a score of 0.0
        :param kwargs:
        """

        self.prefix = prefix.replace(" ", "_")
        assert method in ["max", "mean"]
        self.thresh = thresh
        self.method = method
        self.n_jobs = n_jobs
        self.mapper = Pool(n_jobs, return_map=True)
        self.timeout = timeout

        # If file path provided, load smiles.
        if isinstance(ref_smiles, str):
            with open(ref_smiles, "r") as f:
                self.ref_smiles = f.read().splitlines()
        else:
            assert isinstance(ref_smiles, list) and (
                len(ref_smiles) > 0
            ), "None list or empty list provided"
            self.ref_smiles = ref_smiles

        # Canonicalize ref smiles
        self.ref_smiles = [canonize_smiles(smi) for smi in self.ref_smiles]
        if any([smi is None for smi in self.ref_smiles]):
            raise ValueError("One or more reference smiles could not be canonicalized")

    @staticmethod
    def calculate_sim(
        smi: str,
        ref_smiles: list,
        thresh: float,
        method: str,
        prefix: str,
    ):
        """
        Calculate the Tanimoto coefficient given a SMILES string and list of
         reference fps
        :param smi: SMILES string
        :param ref_smiles: list of reference smiles
        :param thresh: If provided check if similarity is above threshold binarising the similarity coefficients
        :param method: 'mean' or 'max'
        :return: (SMILES, Normalized Levenshtein similarity)
        """
        sim_vec = []
        if smi:
            for ref in ref_smiles:
                sim = max(0, 1 - (levenshtein(smi, ref) / len(ref)))
                sim_vec.append(sim)

            if thresh:
                sim_vec = [sim >= thresh for sim in sim_vec]

            if method == "mean":
                sim = float(np.mean(sim_vec))
            elif method == "max":
                sim = float(np.max(sim_vec))
            else:
                raise ValueError(f"Method {method} not recognised")
        else:
            sim = 0.0
            sim_vec = []

        result = {"smiles": smi, f"{prefix}_Sim": sim}
        result.update({f"{prefix}_Cmpd{i+1}_Sim": sim for i, sim in enumerate(sim_vec)})

        return result

    def _score(self, smiles: list, **kwargs):
        """
        Calculate scores for Tanimoto given a list of SMILES.
        :param smiles: List of SMILES strings
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        calculate_sim_p = partial(
                    self.calculate_sim,
                    ref_smiles=self.ref_smiles,
                    thresh=self.thresh,
                    method=self.method,
                    prefix=self.prefix,
                )
        results = [result for result in self.mapper(calculate_sim_p, smiles)]
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
