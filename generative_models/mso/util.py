"""
Module with utility functions
"""
from rdkit import Chem


def canonicalize_smiles(sml):
    """
    Function that canonicalize a given SMILES
    :param sml: input SMILES
    :return: The canonical version of the input SMILES
    """
    mol = Chem.MolFromSmiles(sml)
    if mol is not None:
        sml = Chem.MolToSmiles(mol)
    return sml