import numpy as np
import rdkit.Chem as Chem


def rmsd(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    """
    Calculate the RMSD between two molecules.

    Args:
        mol1: The first molecule.
        mol2: The second molecule.

    Returns:
        The RMSD value between the two molecules.
    """

    if mol1 is None or mol2 is None:
        return np.nan

    mol1_coords = mol1.GetConformer().GetPositions()
    mol2_coords = mol2.GetConformer().GetPositions()

    try:
        rmsd = np.sqrt(np.mean((mol1_coords - mol2_coords) ** 2))
    except Exception as e:
        print(f"Error calculating RMSD: {e}")
        rmsd = np.nan

    return rmsd


def remove_radicals(mol: Chem.Mol, sanitize: bool = True) -> Chem.Mol:
    """Remove free radicals from a molecule."""

    # Iterate over all atoms in the molecule
    for atom in mol.GetAtoms():
        # Check if the current atom has any radical electrons
        num_radicals = atom.GetNumRadicalElectrons()
        if num_radicals > 0:
            # Saturate the atom with hydrogen atoms
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + num_radicals)
            atom.SetNumRadicalElectrons(0)

    if sanitize:
        Chem.SanitizeMol(mol)

    return mol


def has_radicals(mol: Chem.Mol) -> bool:
    """Check if a molecule has any free radicals."""

    # Iterate over all atoms in the molecule
    for atom in mol.GetAtoms():
        # Check if the current atom has any radical electrons
        num_radicals = atom.GetNumRadicalElectrons()
        if num_radicals > 0:
            return True

    return False
