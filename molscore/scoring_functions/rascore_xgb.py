"""
Adapted from https://github.com/reymond-group/RAscore published https://doi.org/10.1039/d0sc05401a
"""

import os
import numpy as np
import pickle as pkl
import importlib_resources as resources
from multiprocessing import Pool


from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.DataStructs import cDataStructs


class RAScore_XGB:
    """
    Predicted synthetic feasibility according to solveability by AiZynthFinder https://doi.org/10.1039/d0sc05401a
    """
    return_metrics = ['pred_proba']

    def __init__(self, prefix: str = 'RAScore', model: str = 'ChEMBL', n_jobs: int = 1, **kwargs):
        """
        :param prefix: Prefix to identify scoring function instance
        :param model: Either ChEMBL, GDB, GDBMedChem [ChEMBL, GDB, GDBMedChem]
        :param n_jobs: Number of python.multiprocessing jobs for multiprocessing of fps
        :param kwargs:
        """
        self.prefix = prefix
        self.n_jobs = n_jobs

        if model == 'ChEMBL':
            with resources.open_binary('molscore.data.models.RAScore.XGB_chembl_ecfp_counts', 'model.pkl') as f:
                self.model = pkl.load(f)
        elif model == 'GDB':
            with resources.open_binary('molscore.data.models.RAScore.XGB_gdbchembl_ecfp_counts', 'model.pkl') as f:
                self.model = pkl.load(f)
        elif model == 'GDBMedChem':
            with resources.open_binary('molscore.data.models.RAScore.XGB_gdbmedechem_ecfp_counts', 'model.pkl') as f:
                self.model = pkl.load(f)
        else:
            raise "Please select from ChEMBL, GDB or GDBMedChem"
            #with open(os.path.abspath(model), 'rb') as f:
            #    self.model = pkl.load(f)

    @staticmethod
    def calculate_fp(smiles):
        """
        Converts SMILES into a counted ECFP6 vector with features.
        :param smiles: SMILES representation of the moelcule of interest
        :type smiles: str
        :return: ECFP6 counted vector with features
        :rtype: np.array
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=False)
            size = 2048
            arr = np.zeros((size,), np.int32)
            for idx, v in fp.GetNonzeroElements().items():
                nidx = idx % size
                arr[nidx] += int(v)
            return arr
        else:
            return None

    def __call__(self, smiles: list, **kwargs):
        """
        Calculate RAScore given a list of SMILES, if a smiles is abberant or invalid,
        should return 0.0

        :param smiles: List of SMILES strings
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        results = [{'smiles': smi, f'{self.prefix}_pred_proba': 0.0} for smi in smiles]
        valid = []
        fps = []
        with Pool(self.n_jobs) as pool:
            [(valid.append(i), fps.append(fp))
             for i, fp in enumerate(pool.imap(self.calculate_fp, smiles))
             if fp is not None]
        probs = self.model.predict_proba(np.asarray(fps).reshape(len(fps), -1))[:, 1]
        for i, prob in zip(valid, probs):
            results[i].update({f'{self.prefix}_pred_proba': prob})

        return results
