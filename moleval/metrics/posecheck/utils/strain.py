from copy import deepcopy

import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import AllChem, rdForceFieldHelpers

from .chem import has_radicals


def _get_ff(mol: Chem.Mol, forcefield: str, conf_id: int = -1):
    """Gets molecular forcefield for input mol according to name
    Args:
        mol: input molecule
        forcefield: forcefield name. One of "UFF", "MMFF94s", "MMFF94s_noEstat"]
        conf_id: conformer id. -1 is used by default
    """
    assert forcefield in [
        "UFF",
        "MMFF94s",
        "MMFF94s_noEstat",
    ], f"Forcefield {forcefield} is not supported"
    if forcefield == "UFF":
        return rdForceFieldHelpers.UFFGetMoleculeForceField(mol, confId=conf_id)

    mp = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, "MMFF94s")
    if forcefield == "MMFF94s_noEstat":
        mp.SetMMFFEleTerm(False)
    return rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, mp, confId=conf_id)


def calculate_energy(
    mol: Chem.Mol, forcefield: str = "UFF", add_hs: bool = True
) -> float:
    """
    Calculates the energy of a molecule using a force field.

    Args:
        mol: RDKit Mol object representing the molecule.
        forcefield: Force field to use for energy calculation (default: "UFF").
        add_hs: Whether to add hydrogens to the molecule (default: True).

    Returns:
        energy: Calculated energy of the molecule (rounded to 2 decimal places).
                Returns NaN if energy calculation fails.
    """
    mol = Chem.Mol(mol)  # Make a deep copy of the molecule

    if add_hs:
        mol = Chem.AddHs(mol, addCoords=True)

    try:
        ff = _get_ff(mol, forcefield=forcefield)
    except Exception:
        return np.nan

    energy = ff.CalcEnergy()

    return round(energy, 2)


def relax_mol(mol: Chem.Mol) -> Chem.Mol:
    """Relax a molecule by adding hydrogens, embedding it, and optimizing it.

    Args:
        mol (Chem.Mol): The molecule to relax.

    Returns:
        Chem.Mol: The relaxed molecule.
    """

    # if the molecule is None, return None
    if mol is None:
        return None

    # Incase ring info is not present
    Chem.GetSSSR(mol)  # SSSR: Smallest Set of Smallest Rings

    # make a copy of the molecule
    mol = deepcopy(mol)

    # add hydrogens
    mol = Chem.AddHs(mol, addCoords=True)

    # embed the molecule
    AllChem.EmbedMolecule(mol, randomSeed=0xF00D)

    # optimize the molecule
    AllChem.UFFOptimizeMolecule(mol)

    # return the molecule
    return mol


def get_strain_energy(mol: Chem.Mol) -> float:
    """
    Calculates the strain energy of a molecule.

    Strain energy is defined as the difference between the energy of a molecule
    and the energy of the same molecule in a relaxed geometry.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule.

    Returns
    -------
    float
        Strain energy.

    """

    # Check if the molecule has radicals
    assert not has_radicals(
        mol
    ), "Molecule has radicals, consider removing them first. (`posecheck.utils.chem.remove_radicals()`)"

    try:
        return calculate_energy(mol) - calculate_energy(relax_mol(mol))
    except Exception:
        return np.nan


if __name__ == "__main__":
    from posecheck.utils.constants import EXAMPLE_LIGAND_PATH, EXAMPLE_PDB_PATH
    from posecheck.utils.loading import load_mols_from_sdf, load_protein_from_pdb

    # Example molecules
    prot = load_protein_from_pdb(EXAMPLE_PDB_PATH)
    lig = load_mols_from_sdf(EXAMPLE_LIGAND_PATH)[0]

    # Calculate strain
    strain = get_strain_energy(lig)

    print(f"Strain energy: {strain}")

    assert round(strain, 2) == 19.11
