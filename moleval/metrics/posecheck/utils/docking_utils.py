import os
from typing import Dict

import numpy as np
import openbabel
import openbabel as ob
import rdkit.Chem as Chem

#### AutoDock Vina Parsing Functions ####


def parse_smina_output_minimize(out) -> Dict[str, float]:
    """Parse the output of smina --minimize.

    Parameters
    ----------
    out : str
        The output of smina --minimize.

    Returns
    -------
    dict of str: float
    """

    lines = out.split("\n")

    affinity = float(lines[-5].split(" ")[1])
    rmsd = float(lines[-4].split(" ")[1])

    return {"min_affinity": affinity, "min_rmsd": rmsd}


def parse_smina_output_score(out: str) -> dict:
    """
    Parses the output of smina and returns the affinity score, intramolecular energy and energy terms.

    Arguments:
        out: The output of smina as a string.

    Returns:
        A dictionary containing the affinity score, intramolecular energy and energy terms.
    """
    # Split the output into lines
    lines = out.split("\n")

    # Extract the affinity score
    affinity = float(lines[-7].split(" ")[1])

    # Extract the intramolecular energy
    intramolecular_energy = float(lines[-6].split(" ")[-1])

    # Extract the energy terms
    energy_terms = [float(lines[-4].split(" ")[i + 2]) for i in range(6)]

    return {
        "affinity": affinity,
        "intramolecular_energy": intramolecular_energy,
        "energy_terms": energy_terms,
    }


def parse_smina_output_docking(out: str) -> dict:
    """
    Parses Smina output

    Parameters
    ----------
    out : str
        Smina output string

    Returns
    -------
    dict
        Dictionary containing the affinities
    """
    # Check if there is an output
    if "-----+------------+----------+----------" not in out:
        return None

    # Split output into lines
    out_split = out.splitlines()
    # Find the line with the best affinity
    best_idx = out_split.index("-----+------------+----------+----------") + 1

    # Get all of the affinities
    affinities = []

    # Loop poses until the 'Refine' line is reached
    while True:
        line = out_split[best_idx]
        if line.split()[0] == "Refine":
            break
        else:
            affinities.append(float(line.split()[1]))
            best_idx += 1

    return {"affinities": affinities}


#### Open Babel Parsing Functions For Receptor Centroid ####


def read_pdbqt_receptor(file_path):
    """Read a PDBQT file and return an Open Babel molecule object.
    This is done to get the centroid of the receptor.
    """
    # Create an Open Babel molecule object
    ob_mol = openbabel.OBMol()

    # Create an Open Babel file format object for PDBQT
    file_format = openbabel.OBConversion()
    file_format.SetInAndOutFormats("pdbqt", "pdbqt")

    # Read the PDBQT file
    if file_format.ReadFile(ob_mol, file_path):
        return ob_mol
    else:
        raise ValueError(f"Error reading PDBQT file: {file_path}")


def get_coordinates_ob(molecule):
    coordinates = []
    for atom in openbabel.OBMolAtomIter(molecule):
        x, y, z = atom.GetX(), atom.GetY(), atom.GetZ()
        coordinates.append((x, y, z))
    return coordinates


def get_centroid_ob(molecule):
    """Get centroid of molecule"""
    coordinates = get_coordinates_ob(molecule)
    centroid = np.mean(coordinates, axis=0)
    return centroid


def get_centroid_rdmol(mol: Chem.Mol) -> np.ndarray:
    """Calculate the centroid of a molecule.

    Args:
        mol (Chem.Mol): RDKit molecule object.

    Returns:
        np.ndarray: The centroid of the molecule.
    """
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    centroid = np.mean(positions, axis=0)
    return centroid


def update_coordinates(mol, new_coords):
    """
    Update the coordinates of an RDKit molecule.

    Parameters:
    - mol: RDKit molecule object
    - new_coords: 2D list or array of shape (num_atoms, 3) containing the new coordinates

    Returns:
    - RDKit molecule with updated coordinates
    """

    if not mol.GetNumConformers():
        raise ValueError("The molecule has no conformers.")

    conf = mol.GetConformer()

    for idx, (x, y, z) in enumerate(new_coords):
        conf.SetAtomPosition(idx, (x, y, z))

    return mol


#### Read and Write PDBQT Files to RDKIT Mols ####


def rdkit_mol_to_pdbqt(rdkit_mol, pdbqt_path):
    """Save an RDKit molecule to a PDBQT file using an intermediate PDB file."""

    # Create a temporary PDB file path with a random name
    temp_pdb_path = f"temp_molecule_{np.random.randint(1000000)}.pdb"

    # print(rdkit_mol.GetConformer().GetPositions()[:4])
    # TODO REMOVE

    # AllChem.EmbedMolecule(rdkit_mol, AllChem.ETKDG())
    # ! NOTE careful! this added random coords

    # Write RDKit molecule to the temporary PDB file
    writer = Chem.PDBWriter(temp_pdb_path)
    writer.write(rdkit_mol)
    writer.close()

    # Set up Open Babel objects
    mol = ob.OBMol()
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("pdb", "pdbqt")

    # Read the temporary PDB file with Open Babel
    if not obConversion.ReadFile(mol, temp_pdb_path):
        raise ValueError(f"Failed to read the temporary PDB file: {temp_pdb_path}")

    # Write the molecule to a PDBQT file
    if not obConversion.WriteFile(mol, pdbqt_path):
        raise ValueError(f"Failed to write to the PDBQT file: {pdbqt_path}")

    # Clean up temporary PDB file
    os.remove(temp_pdb_path)


def pdbqt_to_rdkit_mols(pdbqt_path):
    """Convert all molecules in a PDBQT file to a list of RDKit molecules."""

    # Set up Open Babel objects
    mol = ob.OBMol()
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("pdbqt", "pdb")

    # Ensure the file can be opened
    if not obConversion.OpenInAndOutFiles(pdbqt_path, ""):
        raise ValueError(f"Failed to read the PDBQT file: {pdbqt_path}")

    rdkit_mols = []
    # Loop over all molecules in the file
    not_end_of_file = True
    while not_end_of_file:
        not_end_of_file = obConversion.Read(mol)
        # Convert to PDB format
        pdb_block = obConversion.WriteString(mol)

        # Convert PDB block to RDKit molecule
        rdkit_mol = Chem.MolFromPDBBlock(pdb_block)

        if rdkit_mol is not None:
            rdkit_mols.append(rdkit_mol)

        # Clear the molecule object for the next iteration
        mol.Clear()

    if not rdkit_mols:
        raise ValueError("No molecules could be successfully converted.")

    # Remove the last molecule, which is always empty
    return rdkit_mols[:-1]
