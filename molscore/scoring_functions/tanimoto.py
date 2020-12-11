import logging
import numpy as np
from functools import partial
from multiprocessing import Pool
from rdkit.Chem import AllChem as Chem

logger = logging.getLogger('tanimoto')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class TanimotoSimilarity:
    """Scoring function class to score structures based on Tanimoto similarity to reference structures."""
    def __init__(self, prefix: str, ref_smiles: [list, str],
                 radius: int = 2, bits: int = 1024, features: str = False,
                 method: str = 'mean', n_jobs: int = 1, **kwargs):
        """
        Scoring function class to score structures based on Tanimoto similarity to reference structures.
        :param prefix: Name (to help keep track metrics, if using a scoring function class more than once)
        :param ref_smiles: List of SMILES strings for reference structures
        :param radius: Radius of Morgan fingerprints (default 2)
        :param bits: Number of Morgan fingerprint bits (default 1024)
        :param features: Whether to include feature information (FCFP) (default False)
        :param method: Either calculate the 'mean' Tanimoto similarity to ref structures or 'max' Tanimoto similarity
         of ref structures. (default mean)
        :param n_jobs: Number of jobs for multiprocessing
        :param kwargs: Ignored
        """
        self.prefix = prefix.replace(" ", "_")
        assert method in ['max', 'mean']
        self.method = method
        self.score_metrics = [f'{method}_Tc']
        self.radius = radius
        self.bits = bits
        self.features = features
        self.n_jobs = n_jobs

        # If file path provided, load smiles.
        if isinstance(ref_smiles, str):
            with open(ref_smiles, 'r') as f:
                self.ref_smiles = f.read().splitlines()
        else:
            assert isinstance(ref_smiles, list) and (len(ref_smiles) > 0), "Non list or empty list provided"
            self.ref_smiles = ref_smiles

        # Convert ref smiles to mols
        self.ref_mols = [Chem.MolFromSmiles(smi) for smi in self.ref_smiles
                           if Chem.MolFromSmiles(smi)]
        # Check they're the same length
        if len(self.ref_smiles) != len(self.ref_mols):
            logger.warning(f"{len(self.ref_smiles) - len(self.ref_mols)} query smiles converted to mol successfully")
        # Conver ref mols to ref fps
        self.ref_fps = [Chem.GetMorganFingerprintAsBitVect(mol, radius=self.radius, nBits=self.bits,
                                                           useFeatures=self.features)
                        for mol in self.ref_mols]

    @staticmethod
    def calculate_Tc(smi: str, ref_fps: np.ndarray, radius: int, nBits: int,
        useFeatures: bool, method: str):
        """
        Calculate the Tanimoto coefficient given a SMILES string and list of
         reference fps (np.array for multiprocessing, calculating
          Tc using numpy was faster than converting back to ExplicitBitVect).
        :param smi: SMILES string
        :param ref_fps: ndarray of reference bit vectors
        :param radius: Radius of Morgan fingerprints
        :param nBits: Number of Morgan fingerprint bits
        :param useFeatures: Whether to include feature information
        :param method: 'mean' or 'max'
        :return: (SMILES, Tanimoto coefficient)
        """

        # Calculate similarity using numpy to allow parallel processing
        # this is faster than converting np back to bit vector (x5), but not as fast as BulkTanimotoSimilarity (x10)
        def np_tanimoto(v1, v2):
            return np.bitwise_and(v1, v2).sum() / np.bitwise_or(v1, v2).sum()

        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = np.array(Chem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits,
                                                             useFeatures=useFeatures))
            Tc_vec = [np_tanimoto(fp, rfp) for rfp in ref_fps]
            if method == 'mean':
                Tc = np.mean(Tc_vec)
            elif method == 'max':
                Tc = np.max(Tc_vec)
            else:
                Tc = 0.0
        else:
            Tc = 0.0

        return smi, Tc

    def __call__(self, smiles: list, **kwargs):
        """
        Calculate scores for Tanimoto given a list of SMILES.
        :param smiles: List of SMILES strings
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        with Pool(self.n_jobs) as pool:
            calculate_Tc_p = partial(self.calculate_Tc, ref_fps=np.asarray(self.ref_fps), radius=self.radius,
                                     nBits=self.bits, useFeatures=self.features, method=self.method)
            results = [{'smiles': smi, f'{self.prefix}_{self.method}_Tc': Tc}
                       for smi, Tc in pool.imap(calculate_Tc_p, smiles)]
        return results
