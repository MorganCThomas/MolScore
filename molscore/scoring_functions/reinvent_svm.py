"""
Adapted from
https://github.com/MarcusOlivecrona/REINVENT
"""

from __future__ import division, print_function

import os

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from sklearn.externals import joblib

rdBase.DisableLog("rdApp.error")


class ActivityModel:
    """This particular class uses the SVM taken from the REINVENT publication and
    refactors code used for fingerprint generation.
    https://github.com/MarcusOlivecrona/REINVENT
    """

    return_metrics = ["pred_proba"]

    def __init__(self, prefix: str, model_path: os.PathLike, **kwargs):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param model_path: Path to pre-trained model (specifically clf.pkl in REINVENT publication)
        :param kwargs:
        """
        self.clf_path = model_path
        self.prefix = prefix.replace(" ", "_")
        self.clf = joblib.load(self.clf_path)

    def __call__(self, smiles, **kwargs):
        """
        Calculate scores for ActivityModel given a list of SMILES
        :param smiles: List of SMILES strings
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        # If just a single str
        if isinstance(smiles, str):
            results = {"smiles": smiles}
            mol = Chem.MolFromSmiles(smiles)

            if mol:
                fp = ActivityModel.fingerprints_from_mol(mol)
                score = self.clf.predict_proba(fp)[:, 1]
                results.update({f"{self.prefix}_pred_proba": float(score[0])})
            else:
                results.update({f"{self.prefix}_pred_proba": 0.0})
            return results

        elif isinstance(smiles, list):
            results = []
            valid = []
            fps = []

            # Calculate fingerprints
            for i, smi in enumerate(smiles):
                result = {"smiles": smi}
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    fp = ActivityModel.fingerprints_from_mol(mol)
                    fps.append(fp.reshape(-1))
                    valid.append(i)
                else:
                    result.update({f"{self.prefix}_pred_proba": 0.0})
                results.append(result)

            # Grab prediction
            y_prob = self.clf.predict_proba(np.asarray(fps))[:, 1]
            for i, prob in zip(valid, y_prob):
                results[i].update({f"{self.prefix}_pred_proba": prob})

            return results

        else:
            print("smiles not provided in correct format")
            raise

    @classmethod
    def fingerprints_from_mol(cls, mol: Chem.rdchem.Mol):
        """
        Calculate folded bit fingerprint using count and features.
        :param mol: rdkit mol
        :return: fp (ndarray)
        """
        fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
        size = 2048
        nfp = np.zeros((1, size), np.int32)
        for idx, v in fp.GetNonzeroElements().items():
            nidx = idx % size
            nfp[0, nidx] += int(v)
        return nfp
