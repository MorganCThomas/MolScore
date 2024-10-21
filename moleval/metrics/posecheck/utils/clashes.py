from typing import List, Union

import numpy as np
import rdkit.Chem as Chem

from .chem import has_radicals


def count_clashes(lig: Chem.Mol, prot: Chem.Mol, tollerance: float = 0.5) -> int:
    """
    Counts the number of clashes between atoms in a protein and a ligand.

    Args:
        prot: RDKit Mol object representing the protein.
        lig: RDKit Mol object representing the ligand.
        tolerance: Distance tolerance for clash detection (default: 0.5).

    Returns:
        clashes: Number of clashes between the protein and the ligand.
    """

    # Check if the molecule has radicals
    assert not has_radicals(
        lig
    ), "Molecule has radicals, consider removing them first. (`posecheck.utils.chem.remove_radicals()`)"

    clashes = 0

    try:
        # Get the positions of atoms in the protein and ligand
        prot_pos = prot.GetConformer().GetPositions()
        lig_pos = lig.GetConformer().GetPositions()

        pt = Chem.GetPeriodicTable()

        # Get the number of atoms in the protein and ligand
        num_prot_atoms = prot.GetNumAtoms()
        num_lig_atoms = lig.GetNumAtoms()

        # Calculate the Euclidean distances between all atom pairs in the protein and ligand
        dists = np.linalg.norm(
            prot_pos[:, np.newaxis, :] - lig_pos[np.newaxis, :, :], axis=-1
        )

        # Iterate over the ligand atoms
        for i in range(num_lig_atoms):
            lig_vdw = pt.GetRvdw(lig.GetAtomWithIdx(i).GetAtomicNum())

            # Iterate over the protein atoms
            for j in range(num_prot_atoms):
                prot_vdw = pt.GetRvdw(prot.GetAtomWithIdx(j).GetAtomicNum())

                # Check for clash by comparing the distances with tolerance
                if dists[j, i] + tollerance < lig_vdw + prot_vdw:
                    clashes += 1

    except AttributeError:
        raise ValueError(
            "Invalid input molecules. Please provide valid RDKit Mol objects."
        )

    return clashes


#### Helper functions


def count_clashes_list(
    prot: Chem.Mol,
    ligs: List[Union[Chem.Mol, str]],
    target: Union[str, None] = None,
    tollerance: float = 0.5,
) -> List[dict]:
    """
    Counts the number of clashes between atoms in a protein and a list of ligands.

    Args:
        prot: RDKit Mol object representing the protein.
        ligs: List of RDKit Mol objects or SMILES strings representing the ligands.
        target: Target identifier associated with the ligands (default: None).
        tolerance: Distance tolerance for clash detection (default: 0.5).

    Returns:
        clashes: List of dictionaries containing clash information for each ligand.
                 Each dictionary contains the following keys: 'mol' (RDKit Mol object),
                 'clashes' (number of clashes), and 'target' (target identifier).
    """
    clashes = []

    # Iterate over the ligands
    for lig in ligs:
        try:
            # Create RDKit Mol object from SMILES string or existing Mol object
            if isinstance(lig, str):
                lig = Chem.MolFromSmiles(lig)
            else:
                lig = Chem.Mol(lig)

            # Count clashes between protein and ligand
            lig_clashes = count_clashes(prot, lig, tollerance)

            # Append clash information to the clashes list
            clashes.append({"mol": lig, "clashes": lig_clashes, "target": target})
        except Exception:
            # Handle errors by appending a dictionary with NaN clashes
            clashes.append({"mol": lig, "clashes": np.nan, "target": target})

    return clashes


if __name__ == "__main__":
    from posecheck.utils.constants import EXAMPLE_LIGAND_PATH, EXAMPLE_PDB_PATH
    from posecheck.utils.loading import load_mols_from_sdf, load_protein_from_pdb

    # Example molecules
    prot = load_protein_from_pdb(EXAMPLE_PDB_PATH)
    lig = load_mols_from_sdf(EXAMPLE_LIGAND_PATH)[0]

    # Count clashes between protein and ligand
    clashes = count_clashes(prot, lig)

    # Print the number of clashes
    print(f"Number of clashes: {clashes}")

    assert clashes == 2
