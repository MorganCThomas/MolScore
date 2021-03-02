from multiprocessing import Pool
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
                 fp_type: str, **kwargs):
        self.prefix = prefix
        self.prefix = prefix.replace(" ", "_")
        self.model_path = model_path
        self.fp_type = fp_type

        # Load in model and assign to attribute
        self.model = joblib.load(model_path)

    @staticmethod
    def calculate_fp(smiles):
        return fp

    def __call__(self, smiles: list, **kwargs):
        """
        Calculate scores for an sklearn model given a list of SMILES, if a smiles is abberant or invalid,
         should return 0.0 for all metrics for that smiles

        :param smiles: List of SMILES strings
        :param kwargs: Ignored
        :return: List of dicts i.e. [{'smiles': smi, 'metric': 'value', ...}, ...]
        """
        return results