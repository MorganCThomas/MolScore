from multiprocessing import Pool
from functools import partial
import glob
import os
import rdkit
import numpy as np
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem
from rdkit.Avalon import pyAvalonTools
from rdkit import rdBase
import joblib

from molscore.scoring_functions.utils import Fingerprints
rdBase.DisableLog('rdApp.error')


class SKLearnModel:
    """
    Score structures by loading a pre-trained sklearn model and return the predicted values
    """
    return_metrics = ['pred_proba']

    def __init__(self, prefix: str, model_path: os.PathLike,
                 fp: str, nBits: int = 1024, n_jobs: int = 1, **kwargs):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param model_path: Path to pre-trained model (saved using joblib)
        :param fp: What type of fingerprint to use [ECFP4, ECFP4c, FCFP4, FCFP4c, ECFP6, ECFP6c, FCFP6, FCFP6c, Avalon, MACCSkeys, hashAP, hashTT, RDK5, RDK6, RDK7]
        :param nBits: Length of fingerprint
        :param n_jobs: Number of python.multiprocessing jobs for multiprocessing of fps
        :param kwargs:
        """
        self.prefix = prefix
        self.prefix = prefix.replace(" ", "_")
        self.model_path = model_path
        self.fp_type = fp
        self.nBits = int(nBits)
        self.n_jobs = n_jobs

        # Load in model and assign to attribute
        self.model = joblib.load(model_path)

    @staticmethod
    def calculate_fp(smi: str, fp_type: str, nBits=1024):
        """Calculates fp based on fp_type and smiles"""
        
        mol = Chem.MolFromSmiles(smi)
        if mol:
            #Circular fingerprints
            if fp_type == "ECFP4":
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nBits) # ECFP4
            elif fp_type == "ECFP6":
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=nBits) # ECFP6

            # Structural fingerprints:
            elif fp_type == "Avalon":
                fp = pyAvalonTools.GetAvalonFP(mol, nBits=nBits) # Avalon
            elif fp_type == "MACCSkeys":
                fp = rdkit.Chem.rdMolDescriptors.GetMACCSKeysFingerprint(mol) #MACCS Keys
            
            # Path-based fingerprints
            elif fp_type == "hashAP":
                fp = rdkit.Chem.rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=nBits)
            elif fp_type == "hashTT":
                fp = rdkit.Chem.rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=nBits)
            elif fp_type == "RDK5":
                fp = rdkit.Chem.rdmolops.RDKFingerprint(mol, maxPath=5, fpSize=nBits, nBitsPerHash=2)
            elif fp_type == "RDK6":
                fp = rdkit.Chem.rdmolops.RDKFingerprint(mol, maxPath=6, fpSize=nBits, nBitsPerHash=2)
            elif fp_type == "RDK7":
                fp = rdkit.Chem.rdmolops.RDKFingerprint(mol, maxPath=7, fpSize=nBits, nBitsPerHash=2)
        
            return np.asarray(fp).reshape(1, -1)

        else:
            return None

    def __call__(self, smiles: list, **kwargs):
        """
        Calculate scores for an sklearn model given a list of SMILES, if a smiles is abberant or invalid,
         should return 0.0 for all metrics for that smiles

        :param smiles: List of SMILES strings
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """

        results = [{'smiles': smi, f'{self.prefix}_pred_proba': 0.0} for smi in smiles]
        valid = []
        fps = []
        with Pool(self.n_jobs) as pool:
            #pcalulate_fp = partial(self.calculate_fp, fp_type=self.fp_type, nBits=self.nBits)
            pcalculate_fp = partial(Fingerprints.get, name=self.fp_type, nBits=self.nBits, asarray=True)
            [(valid.append(i), fps.append(fp.reshape(1, -1)))
             for i, fp in enumerate(pool.imap(pcalculate_fp, smiles))
             if fp is not None]

        if len(valid) != 0 :
            probs = self.model.predict_proba(np.asarray(fps).reshape(len(fps), -1))[:, 1]
            for i, prob in zip(valid, probs):
                results[i].update({f'{self.prefix}_pred_proba': prob})

        return results


class EnsembleSKLearnModel(SKLearnModel):
    """
    This class loads different random seeds of a defined sklearn
    model and returns the average of the predicted values.
    """
    def __init__(self, prefix: str, model_path: os.PathLike,
                 fp_type: str, nBits: int, n_jobs: int = 1, **kwargs):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param model_path: Path to pre-trained model (saved using joblib)
        :param fp_type: 'ECFP4', 'ECFP6', 'Avalon', 'MACCSkeys', 'hashAP', 'hashTT', 'RDK5', 'RDK6', 'RDK7'
        :param nBits: Length of fingerprint
        :param n_jobs: Number of python.multiprocessing jobs for multiprocessing of fps
        :param kwargs:
        """
        super().__init__(prefix, model_path, 
                        fp_type, nBits, n_jobs, **kwargs)
        changing = self.model_path.split('_')
        del changing[len(changing)-1]
        changing = '_'.join(changing)
        self.replicates = sorted(glob.glob(str(changing) + '*.joblib'))

        # Load in model and assign to attribute
        self.models = []
        for f in self.replicates:
            self.models.append(joblib.load(f))
    
    def __call__(self, smiles: list, **kwargs):
        results = [{'smiles': smi, f'{self.prefix}_pred_proba': 0.0} for smi in smiles]
        valid = []
        fps = []
        predictions = []
        averages = []
        with Pool(self.n_jobs) as pool:
            #pcalulate_fp = partial(self.calculate_fp, fp_type=self.fp_type, nBits=self.nBits)
            pcalculate_fp = partial(Fingerprints.get, name=self.fp_type, nBits=self.nBits, asarray=True)
            [(valid.append(i), fps.append(fp))
             for i, fp in enumerate(pool.imap(pcalculate_fp, smiles))
             if fp is not None]
        
        # Predicting the probabilies and appending them to a list.
        for m in self.models:
            prediction = m.predict_proba(np.asarray(fps).reshape(len(fps), -1))[:, 1]
            predictions.append(prediction)
        predictions = np.asarray(predictions)
        averages = predictions.mean(axis=0)

        for i, prob in zip(valid, averages):
            results[i].update({f'{self.prefix}_pred_proba': prob})

        return results
