import random

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
