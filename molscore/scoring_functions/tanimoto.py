import os
import argparse
import logging
import numpy as np
from functools import partial
from multiprocessing import Pool
from rdkit.Chem import AllChem as Chem
from typing import Union

logger = logging.getLogger('tanimoto')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class TanimotoSimilarity:
    """
    Score structures based on Tanimoto similarity of Morgan fingerprints to reference structures
    """
    return_metrics = ['Tc']

    def __init__(self, prefix: str, ref_smiles: Union[list, os.PathLike],
                 radius: int = 2, bits: int = 1024, features: bool = False,
                 counts: bool = False, method: str = 'mean', n_jobs: int = 1, **kwargs):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param ref_smiles: List of SMILES or path to SMILES file with no header (.smi)
        :param radius: Radius of Morgan fingerprints
        :param bits: Number of Morgan fingerprint bits
        :param features: Whether to include feature information (sometimes referred to as FCFP)
        :param counts: Whether to include bit count (sometimes referred to as ECFC)
        :param method: 'mean' or 'max' ('max' is equiv. singler nearest neighbour)
        :param n_jobs: Number of python.multiprocessing jobs for multiprocessing
        :param kwargs:
        """
        self.prefix = prefix.replace(" ", "_")
        assert method in ['max', 'mean']
        self.method = method
        self.radius = radius
        self.bits = bits
        self.features = features
        self.counts = counts
        self.n_jobs = n_jobs

        # If file path provided, load smiles.
        if isinstance(ref_smiles, str):
            with open(ref_smiles, 'r') as f:
                self.ref_smiles = f.read().splitlines()
        else:
            assert isinstance(ref_smiles, list) and (len(ref_smiles) > 0), "None list or empty list provided"
            self.ref_smiles = ref_smiles

        # Convert ref smiles to mols
        self.ref_mols = [Chem.MolFromSmiles(smi) for smi in self.ref_smiles
                         if Chem.MolFromSmiles(smi)]

        # Check they're the same length
        if len(self.ref_smiles) != len(self.ref_mols):
            logger.warning(f"{len(self.ref_mols)}/{len(ref_smiles)} query smiles converted to mol successfully")

        # Convert ref mols to ref fps
        if self.counts:
            self.ref_fps = [Chem.GetMorganFingerprint(mol, radius=self.radius, useFeatures=self.features)
                            for mol in self.ref_mols]
        else:
            self.ref_fps = [Chem.GetMorganFingerprintAsBitVect(mol, radius=self.radius, nBits=self.bits,
                                                               useFeatures=self.features)
                            for mol in self.ref_mols]

    @staticmethod
    def calculate_Tc(smi: str, ref_fps: np.ndarray, radius: int, nBits: int,
        useFeatures: bool, useCounts: bool, method: str):
        """
        Calculate the Tanimoto coefficient given a SMILES string and list of
         reference fps (np.array for multiprocessing, calculating
          Tc using numpy was faster than converting back to ExplicitBitVect).
        :param smi: SMILES string
        :param ref_fps: ndarray of reference bit vectors
        :param radius: Radius of Morgan fingerprints
        :param nBits: Number of Morgan fingerprint bits
        :param useFeatures: Whether to include feature information
        :param useCounts: Whether to include count information
        :param method: 'mean' or 'max'
        :return: (SMILES, Tanimoto coefficient)
        """

        # Calculate similarity using numpy to allow parallel processing
        # this is faster than converting np back to bit vector (x5), but not as fast as BulkTanimotoSimilarity (x10)
        def np_tanimoto(v1, v2):
            return np.bitwise_and(v1, v2).sum() / np.bitwise_or(v1, v2).sum()

        mol = Chem.MolFromSmiles(smi)
        if mol:
            if useCounts:
                fp = Chem.GetMorganFingerprint(mol, radius=radius, useFeatures=useFeatures)
                Tc_vec = [Chem.DataStructs.TanimotoSimilarity(fp, rfp) for rfp in ref_fps]
            else:
                fp = np.asarray(Chem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits,
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
                                     nBits=self.bits, useFeatures=self.features, useCounts=self.counts,
                                     method=self.method)
            results = [{'smiles': smi, f'{self.prefix}_Tc': Tc}
                       for smi, Tc in pool.imap(calculate_Tc_p, smiles)]
        return results


if __name__ == '__main__':
    # Read in CLI arguments
    parser = argparse.ArgumentParser(description='Calculate the Maximum or Average Tanimoto similarity for a set of '
                                                 'molecules to a set of reference molecules',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(title='Running mode', dest='mode')
    run = subparsers.add_parser('run', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    run.add_argument('--prefix', type=str, default='test', help='Prefix to returned metrics')
    run.add_argument('--ref_smiles', help='Path to smiles file (.smi) for ref smiles')
    run.add_argument('--query_smiles', help='Path to smiles file (.smi) for query smiles')
    run.add_argument('--radius', default=2, help='Morgan fingerprint radius')
    run.add_argument('--bits', default=1024, help='Morgan fingerprint bit size')
    run.add_argument('--features', action='store_true', help='Use Morgan fingerprint features')
    run.add_argument('--counts', action='store_true', help='Use Morgan fingerprint features')
    run.add_argument('--method', type=str, choices=['mean', 'max'], default='mean',
                     help='Use mean or max similarity to ref smiles')
    run.add_argument('--n_jobs', default=1, type=int, help='How many cores to use')
    test = subparsers.add_parser('test')
    args = parser.parse_args()

    # Run mode
    if args.mode == 'run':
        from molscore.test import MockGenerator
        mg = MockGenerator(seed_no=123)

        if args.ref_smiles is None:
            args.ref_smiles = mg.sample(10)

        ts = TanimotoSimilarity(prefix=args.prefix, ref_smiles=args.ref_smiles, radius=args.radius,
                                bits=args.bits, features=args.features, counts=args.counts,
                                method=args.method, n_jobs=args.n_jobs)

        if args.query_smiles is None:
            _ = [print(o) for o in ts(mg.sample(5))]
        else:
            with open(args.query_smiles, 'r') as f:
                qsmiles = f.read().splitlines()
            _ = [print(o) for o in ts(qsmiles)]

    # Test mode
    elif args.mode == 'test':
        import sys
        import unittest
        from molscore.test.tests import test_tanimoto
        # Remove CLI arguments
        for i in range(len(sys.argv)-1):
            sys.argv.pop()
        # Get and run tests
        suite = unittest.TestLoader().loadTestsFromModule(test_tanimoto)
        unittest.TextTestRunner().run(suite)

    # Print usage
    else:
        print('Please specify running mode: \'run\' or \'test\'')
