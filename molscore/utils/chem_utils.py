import random
from typing import Union

from rdkit.Chem import AllChem as Chem


def randomize_smiles(smi, random_type="restricted"):
    """
    Returns a random SMILES given a SMILES of a molecule.
    :param smi: A SMILES string
    :param random_type: The type (unrestricted, restricted) of randomization performed.
    :return : A randomized version of the SMILES string.
    """
    assert random_type in [
        "restricted",
        "unrestricted",
    ], f"Type {random_type} is not valid"

    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return smi

    if random_type == "unrestricted":
        random_smiles = Chem.MolToSmiles(
            mol, canonical=False, doRandom=True, isomericSmiles=True
        )

        return random_smiles

    if random_type == "restricted":
        new_atom_order = list(range(mol.GetNumAtoms()))
        random.shuffle(new_atom_order)
        random_mol = Chem.RenumberAtoms(mol, newOrder=new_atom_order)
        random_smiles = Chem.MolToSmiles(
            random_mol, canonical=False, isomericSmiles=True
        )

        return random_smiles


def augment_smiles(smiles: list, random_type="restricted") -> list:
    """
    Augments a list of SMILES by randomizing the SMILES strings.
    :param smiles: A list of SMILES strings.
    :param random_type: The type (unrestricted, restricted) of randomization performed.
    :return : A list of randomized SMILES strings.
    """
    return [randomize_smiles(smi, random_type) for smi in smiles]


def canonicalize_smiles(smiles: Union[str, list]) -> list:
    """
    Canonicalizes a list of SMILES strings. If failed, original smiles are returned.
    :param smiles: A list of SMILES strings.
    :return : A list of canonicalized SMILES strings.
    """
    if isinstance(smiles, str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        else:
            return Chem.MolToSmiles(mol)
    if isinstance(smiles, list):
        can_smiles = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                can_smiles.append(smi)
            else:
                can_smiles.append(Chem.MolToSmiles(mol))
        return can_smiles
