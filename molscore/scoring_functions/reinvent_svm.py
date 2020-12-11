"""
Adapted from
https://github.com/MarcusOlivecrona/REINVENT
"""
from __future__ import print_function, division
import numpy as np
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
from sklearn.externals import joblib
rdBase.DisableLog('rdApp.error')


class ActivityModel:
    """ This particular class uses the SVM taken from the REINVENT publication and
     refactors code used for fingerprint generation.
     https://github.com/MarcusOlivecrona/REINVENT
    """

    def __init__(self, prefix: str, cpath: str, **kwargs):
        """
        This particular class uses the SVM taken from the REINVENT publication and
             refactors code used for fingerprint generation.
                  https://github.com/MarcusOlivecrona/REINVENT
        :param prefix: Name (to help keep track metrics, if using a scoring function class more than once)
        :param cpath: File path to scikit-learn model (probably only works with reinvent clf.pkl)
        :param kwargs: Ignored
        """
        self.clf_path = cpath
        self.prefix = prefix.replace(" ", "_")
        self.clf = joblib.load(self.clf_path)
        self.score_metrics = ['pred_prob']

    def __call__(self, smiles, **kwargs):
        """
        Calculate scores for ActivityModel given a list of SMILES
        :param smiles: List of SMILES strings
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        # If just a single str
        if isinstance(smiles, str):
            results = {'smiles': smiles}
            mol = Chem.MolFromSmiles(smiles)

            if mol:
                fp = ActivityModel.fingerprints_from_mol(mol)
                score = self.clf.predict_proba(fp)[:, 1]
                results.update({f'{self.prefix}_pred_prob': float(score[0])})
            else:
                results.update({f'{self.prefix}_pred_prob': 0.0})
            return results

        elif isinstance(smiles, list):
            results = []
            valid = []
            fps = []

            # Calculate fingerprints
            for i, smi in enumerate(smiles):
                result = {'smiles': smi}
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    fp = ActivityModel.fingerprints_from_mol(mol)
                    fps.append(fp.reshape(-1))
                    valid.append(i)
                else:
                    result.update({f'{self.prefix}_pred_prob': 0.0})
                results.append(result)

            # Grab prediction
            y_prob = self.clf.predict_proba(np.asarray(fps))[:, 1]
            for i, prob in zip(valid, y_prob):
                results[i].update({f'{self.prefix}_pred_prob': prob})

            return results

        else:
            print('smiles not provided in correct format')
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

