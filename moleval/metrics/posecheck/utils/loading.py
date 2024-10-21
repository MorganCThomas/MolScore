import os
import pickle
import subprocess
from typing import List, Union

import MDAnalysis as mda
import openbabel as ob
import prolif as plf
import rdkit.Chem as Chem

from .chem import remove_radicals
from .constants import REDUCE_PATH, SPLIT_PATH


def read_sdf(sdf_path: str) -> List[Chem.Mol]:
    """Read an SDF file and return a list of RDKit molecules.

    Args:
        sdf_path (str): The path to the SDF file.

    Returns:
        List[Chem.Mol]: The list of RDKit molecules.
    """
    with Chem.SDMolSupplier(sdf_path) as suppl:
        mols = list(suppl)
    return mols


def to_sdf(mols: List[Chem.Mol], sdf_path: str):
    """Write a list of RDKit molecules to an SDF file.

    Args:
        mols (List[Chem.Mol]): The list of RDKit molecules.
        sdf_path (str): The path to the SDF file.
    """
    with Chem.SDWriter(sdf_path) as w:
        for mol in mols:
            w.write(mol)
        w.flush()
        w.close()


def load_splits_crossdocked(
    split_path: str = SPLIT_PATH,
):
    """Load the train, val, and test splits."""
    with open(split_path, "rb") as f:
        data = pickle.load(f)
    return data


def get_ids_to_pockets(
    split_path: str = SPLIT_PATH,
):
    """Get the names of the pockets in the CrossDocked test dataset."""
    test_set = load_splits_crossdocked(split_path)["test"]

    pdb_names = [
        pdb_path[0].split("/")[1].split(".")[0].replace("_", "-")
        for pdb_path in test_set
    ]

    return pdb_names


def load_protein_prolif(protein_path: str, rdkit: bool = False):
    """Load protein from PDB file using MDAnalysis
    and convert to plf.Molecule. Assumes hydrogens are present."""
    if rdkit:
        prot = Chem.MolFromPDBFile(protein_path, removeHs=False, sanitize=False)
        prot = plf.Molecule(prot)
    else:
        prot = mda.Universe(protein_path)
        prot = plf.Molecule.from_mda(prot, NoImplicit=False)
    return prot


# @lru_cache(maxsize=100)
def load_protein_from_pdb(
    pdb_path: str, reduce: bool = True, reduce_path: str = REDUCE_PATH
):
    """Load protein from PDB file, add hydrogens, and convert it to a prolif.Molecule.

    Args:
        pdb_path (str): The path to the PDB file.
        reduce_path (str, optional): The path to the reduce executable. Defaults to REDUCE_PATH.

    Returns:
        plf.Molecule: The loaded protein as a prolif.Molecule.
    """
    if reduce:
        tmp_path = pdb_path.split(".pdb")[0] + "_tmp.pdb"

        # Call reduce to make tmp PDB with waters
        reduce_command = f"{reduce_path} -NOFLIP  {pdb_path} -Quiet > {tmp_path}"
        subprocess.run(reduce_command, shell=True)

        # Load the protein from the temporary PDB file
        prot = load_protein_prolif(tmp_path)
        os.remove(tmp_path)
    else:
        prot = load_protein_prolif(pdb_path)

    return prot


def get_pdbqt_mol(pdbqt_block: str) -> Chem.Mol:
    """Convert pdbqt block to rdkit mol by converting with openbabel"""
    # write pdbqt file
    with open("test_pdbqt.pdbqt", "w") as f:
        f.write(pdbqt_block)

    # read pdbqt file from autodock
    mol = ob.OBMol()
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("pdbqt", "pdb")
    obConversion.ReadFile(mol, "test_pdbqt.pdbqt")

    # convert to RDKIT
    mol = Chem.MolFromPDBBlock(obConversion.WriteString(mol))

    # remove tmp file
    os.remove("test_pdbqt.pdbqt")

    return mol


def read_pdbqt(pdbqt_file: str) -> Chem.Mol:
    """Read pdbqt file from autodock and convert to rdkit mol.

    Args:
        pdbqt_file (str): path to pdbqt file

    Returns:
        Chem.Mol: rdkit mol
    """
    with open(pdbqt_file, "r") as f:
        pdbqt_block = f.read()
    return get_pdbqt_mol(pdbqt_block)


def load_sdf_prolif(sdf_path: str) -> plf.Molecule:
    """Load ligand from an SDF file and convert it to a prolif.Molecule.

    Args:
        sdf_path (str): Path to the SDF file.

    Returns:
        plf.Molecule: The loaded ligand as a prolif.Molecule.
    """
    return plf.sdf_supplier(sdf_path)


def load_mols_from_sdf(sdf_path: str, add_hs: bool = True) -> Union[plf.Molecule, List]:
    """Load ligand from an SDF file, add hydrogens, and convert it to a prolif.Molecule.

    Args:
        sdf_path (str): Path to the SDF file.
        add_hs (bool, optional): Whether to add hydrogens. Defaults to True.

    Returns:
        Union[plf.Molecule, List]: The loaded ligand as a prolif.Molecule, or an empty list if no molecule could be loaded.
    """
    tmp_path = sdf_path.split(".sdf")[0] + "_tmp.sdf"

    # Load molecules from the SDF file
    mols = read_sdf(sdf_path)

    # Remove radicals from the molecules
    mols = [remove_radicals(mol) for mol in mols]

    # Add hydrogens to the molecules
    if add_hs:
        mols = [Chem.AddHs(m, addCoords=True) for m in mols if m is not None]

    if len(mols) == 0:
        return []

    # Write the molecules to a temporary file
    to_sdf(mols, tmp_path)
    ligs = load_sdf_prolif(tmp_path)
    os.remove(tmp_path)

    # Turn into list
    ligs = list(ligs)

    return ligs


def load_mols_from_rdkit(
    mol_list: List[Chem.Mol],
    add_hs: bool = True,
) -> Union[plf.Molecule, List]:
    """Load ligand from an RDKit molecule, add hydrogens, and convert it to a prolif.Molecule.

    Processes in the same way as src.utils.loading.load_mols_from_sdf but takes an RDKit molecule as input.

    Args:
        rdkit_mol (Chem.Mol): RDKit molecule.
        add_hs (bool, optional): Whether to add hydrogens. Defaults to True.

    Returns:
        Union[plf.Molecule, List]: The loaded ligand as a prolif.Molecule, or an empty list if no molecule could be loaded.
    """

    # Convert single molecule to list
    if isinstance(mol_list, Chem.Mol):
        print(
            "Converting single molecule to list, to be consistent with load_mols_from_sdf"
        )
        mol_list = [mol_list]

    # Remove radicals from the molecules
    mol_list = [remove_radicals(mol) for mol in mol_list]

    # Add hydrogens to the molecules
    if add_hs:
        mol_list = [Chem.AddHs(m, addCoords=True) for m in mol_list if m is not None]

    # Check if any molecules were loaded
    if len(mol_list) == 0:
        print("No molecules loaded, returning empty list")
        return []

    return [plf.Molecule.from_rdkit(mol) for mol in mol_list]
