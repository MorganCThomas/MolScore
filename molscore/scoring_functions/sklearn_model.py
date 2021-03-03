from multiprocessing import Pool
import rdkit
import numpy as np
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
from sklearn.externals import joblib
rdBase.DisableLog('rdApp.error')

class sklearn_model:
    """
    This class loads a defined sklearn model and returns the predicted values
    """
    def __init__(self, prefix: str, model_path: str,
                 fp_type: str, n_jobs: int, **kwargs):
        self.prefix = prefix
        self.prefix = prefix.replace(" ", "_")
        self.model_path = model_path
        self.fp_type = fp_type
        self.n_jobs = n_jobs

        # Load in model and assign to attribute
        self.model = joblib.load(model_path)

    @staticmethod
    def calculate_fp(smi: str, fp_type: str):
        '''Calculates fp based on fp_type and smiles'''
        
        mol = Chem.MolFromSmiles(smi)
        if mol:
            #Circular fingerprints
            if fp_type == "ECFP4":
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024) # ECFP4
            elif fp_type == "ECFP6":
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024) # ECFP6

            # Structural fingerprints:
            elif fp_type == "Avalon":
                fp = rdkit.Avalon.pyAvalonTools.GetAvalonFP(mol, nBits=1024) # Avalon
            elif fp_type == "MACCSkeys":
                fp = rdkit.Chem.rdMolDescriptors.GetMACCSKeysFingerprint(mol) #MACCS Keys
            
            # Path-based fingerprints
            elif fp_type == "hashAP":
                fp = rdkit.Chem.rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=1024)
            elif fp_type == "hashTT":
                fp = rdkit.Chem.rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=1024)
            elif fp_type == "RDK5":
                fp = rdkit.Chem.rdmolops.RDKFingerprint(mol, maxPath=5, fpSize=1024, nBitsPerHash=2)
            elif fp_type == "RDK6":
                fp = rdkit.Chem.rdmolops.RDKFingerprint(mol, maxPath=6, fpSize=1024, nBitsPerHash=2)
            elif fp_type == "RDK7":
                fp = rdkit.Chem.rdmolops.RDKFingerprint(mol, maxPath=7, fpSize=1024, nBitsPerHash=2)
        
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
        # with Pool(self.n_jobs) as pool:
        #     results = [{'smiles': smi, f'{self.prefix}_{self.model_name}': Tc}
        #                for smi, Tc in pool.imap(parallel_FPs, smiles)]
        results = []
        for smi in smiles:
            result = {'smiles': smi}
            fp = sklearn_model.calculate_fp(smi, self.fp_type)
            if fp is None:
                result.update({f'{self.prefix}_pred_proba': 0.0})
                results.append(result)
                continue
            else:
                probability = self.model.predict_proba(fp)
                result.update({f'{self.prefix}_pred_proba': probability[:, 1][0]})
                results.append(result)
                
        return results