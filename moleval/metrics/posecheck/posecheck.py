import os
from functools import partial
from typing import Callable, List, Union

import pandas as pd
import prolif as plf
from rdkit import Chem
from rdkit.Chem import DataStructs

from moleval.utils import Pool

from .utils.chem import rmsd
from .utils.clashes import count_clashes
from .utils.interactions import generate_interaction_df
from .utils.loading import (
    load_mols_from_rdkit,
    load_mols_from_sdf,
    load_protein_from_pdb,
    load_protein_prolif,
    read_pdbqt,
    to_sdf,
)
from .utils.strain import get_strain_energy


class PoseCheck(object):
    """Main class for PoseCheck.

    Load a protein-ligand complex, check for clashes, strain, interactions, etc.

    Example usage:
    >>> pc = PoseCheck()
    >>> pc.load_protein_from_pdb("data/1a2b.pdb")
    >>> pc.load_ligand_from_sdf("data/1a2b_ligand.sdf")
    # Run posecheck
    >>> pc.check_clashes()
    >>> pc.check_strain()
    >>> pc.check_interactions()
    # Or run all at once
    >>> pc.run()
    """

    def __init__(
        self, reduce_path: str = "reduce", clash_tolerance: float = 0.5, n_jobs=None
    ) -> None:
        """Initialize the PoseCheck class.

        Args:
            reduce_path (str, optional): The path to the reduce executable. Defaults to "reduce".
            clash_tolerance (float, optional): The clash tolerance for checking clashes. Defaults to 0.5 A.
            n_jobs (int, optional): The number of jobs to run in parallel. Defaults to None i.e., serial.
        """
        self.reduce_path = reduce_path
        self.clash_tolerance = clash_tolerance
        self.n_jobs = n_jobs

    def load_protein_from_pdb(self, pdb_path: str, reduce: bool = True) -> None:
        """Load a protein from a PDB file.

        Args:
            pdb_path (str): The path to the PDB file.

        Returns:
            None
        """
        self.pdb_path = pdb_path
        self.protein = load_protein_from_pdb(
            pdb_path, reduce=reduce, reduce_path=self.reduce_path
        )

    def load_ligands_from_sdf(self, sdf_path: str) -> None:
        """Load a ligand from an SDF file."""
        self.ligands = load_mols_from_sdf(sdf_path)

    def load_ligands_from_pdbqt(self, pdbqt_path: str) -> None:
        """Load a ligand from a PDBQT file."""
        mol = read_pdbqt(pdbqt_path)

        # Save to tmp sdf file and load with Hs
        tmp_path = pdbqt_path.split(".pdbqt")[0] + "_tmp.pdb"
        to_sdf(mol, tmp_path)
        self.ligands = load_mols_from_sdf(tmp_path)
        os.remove(tmp_path)

    def load_ligands_from_mols(self, mols: List[Chem.Mol], add_hs: bool = True) -> None:
        """Load ligands from a list of RDKit mols.

        Args:
            mols (List[Chem.Mol]): The list of RDKit mol objects representing the ligands.
            add_hs (bool, optional): Whether to add hydrogens to the ligands. Defaults to True.

        Returns:
            None
        """
        # Can't use parallel mapping or calculate interactions breaks
        self.ligands = load_mols_from_rdkit(mols, add_hs=add_hs)

    def load_ligands(self, ligand) -> None:
        """Detect ligand type and load.

        Args:
            ligand (str or Chem.Mol): The ligand to load.

        Raises:
            ValueError: If the ligand type is unknown.

        Returns:
            None
        """

        if isinstance(ligand, str):
            if ligand.endswith(".sdf"):
                self.load_ligand_from_sdf(ligand)
            elif ligand.endswith(".pdbqt"):
                self.load_ligand_from_pdbqt(ligand)
            else:
                raise ValueError("Unknown ligand type.")
        elif isinstance(ligand, Chem.Mol):
            self.load_ligand_from_mol(ligand)
        else:
            raise ValueError("Unknown ligand type.")

    def calculate_clashes(self) -> int:
        """Calculate the number of steric clashes between protein and ligand."""
        if self.n_jobs:
            with Pool(self.n_jobs) as pool:
                pf = partial(
                    count_clashes, prot=self.protein, tollerance=self.clash_tolerance
                )
                return list(pool.map(pf, self.ligands))
        else:
            return [
                count_clashes(self.protein, mol, tollerance=self.clash_tolerance)
                for mol in self.ligands
            ]

    def calculate_strain_energy(self) -> float:
        """Calculate the strain energy of the ligand."""
        if self.n_jobs:
            with Pool(self.n_jobs) as pool:
                return list(pool.map(get_strain_energy, self.ligands))
        else:
            return [get_strain_energy(mol) for mol in self.ligands]

    def calculate_ref_interactions(self, ref_lig_path: str) -> pd.DataFrame:
        """Calculate the interactions between the protein and the reference ligand"""
        # Reload protein, bug if passed through reduce, bug with MDAnalysis
        # Seems more robust with Chem.MolFromPDB for now...

        # Currently less interactions than repeating the same code in Ipython interactively???
        protein = load_protein_prolif(self.pdb_path, rdkit=True)
        lig = load_mols_from_sdf(ref_lig_path)
        return generate_interaction_df(protein, lig)

    def calculate_interactions(self) -> pd.DataFrame:
        """Calculate the interactions between the protein and the ligand."""
        # Reload protein, bug if passed through reduce, bug with MDAnalysis
        # Seems more robust with Chem.MolFromPDB for now...
        protein = load_protein_prolif(self.pdb_path, rdkit=True)
        return generate_interaction_df(protein, self.ligands)

    def calculate_interaction_similarity(
        self,
        ref_lig_path: str,
        similarity_func: Union[Callable, str],
        count: bool = False,
    ) -> List[float]:
        """Calculate the similarity between the reference ligand and all other ligands."""
        protein = load_protein_prolif(self.pdb_path, rdkit=True)
        ref_lig = load_mols_from_sdf(ref_lig_path)
        residues = plf.utils.get_residues_near_ligand(ref_lig[0], protein, cutoff=6.0)
        # Calculate ref fp
        ref_fp = generate_interaction_df(
            protein, ref_lig, count=count, drop_empty=False, residues=residues
        )
        # Calculate ligand fps
        lig_fps = generate_interaction_df(
            protein, self.ligands, count=count, drop_empty=False, residues=residues
        )
        # Append dataframes to ensure aligned
        ifps = pd.concat([ref_fp, lig_fps], ignore_index=0).fillna(False)
        # Get similarity function
        if callable(similarity_func):
            similarity_function = similarity_func
        else:
            similarity_function = getattr(
                DataStructs, "Bulk" + similarity_func + "Similarity", None
            )
            if similarity_function is None:
                raise KeyError(f"'{similarity_func}' not found in RDKit")
        # Calculate similarity
        bvs = plf.to_bitvectors(ifps)
        similarities = similarity_function(bvs[0], bvs[1:])
        return similarities

    def calculate_rmsd(self, mol1: Chem.Mol, mol2: Chem.Mol) -> float:
        """Calculate the RMSD between two molecules

        For example, the RMSD between the generated ligand pose and one minimized by docking software.
        """
        return rmsd(mol1, mol2)

    def run(self) -> dict:
        """Execute all of PoseCheck suite and return the results.

        This method calculates the clashes, strain energy, and interactions
        between the protein and the ligand. The results are returned as a dictionary.

        Returns:
            dict: A dictionary containing the number of clashes, the strain energy,
                  and the interactions between the protein and the ligand.
        """
        clashes = self.calculate_clashes()
        strain = self.calculate_strain_energy()
        interactions = self.calculate_interactions()

        results = {"clashes": clashes, "strain": strain, "interactions": interactions}
        return results


if __name__ == "__main__":
    from posecheck.utils.constants import EXAMPLE_LIGAND_PATH, EXAMPLE_PDB_PATH

    """Test calculate_clashes method."""
    # Set the protein and ligand in the PoseCheck object
    pc = PoseCheck()
    pc.load_protein_from_pdb(EXAMPLE_PDB_PATH)
    pc.load_ligands_from_sdf(EXAMPLE_LIGAND_PATH)

    # Calculate the clashes
    clashes = pc.calculate_clashes()[0]

    # Check if the clashes are calculated correctly
    assert clashes == 2, "Clashes calculation is incorrect."

    """Test calculate_strain_energy method."""
    # Set the ligand in the PoseCheck object
    pc = PoseCheck()
    pc.load_ligands_from_sdf(EXAMPLE_LIGAND_PATH)

    # Calculate the strain energy
    strain_energy = pc.calculate_strain_energy()[0]

    # Check if the strain energy is calculated correctly
    assert round(strain_energy, 2) == 19.11

    """Test calculate_interactions method."""
    # Set the protein and ligand in the PoseCheck object
    pc = PoseCheck()
    pc.load_protein_from_pdb(EXAMPLE_PDB_PATH)
    pc.load_ligands_from_sdf(EXAMPLE_LIGAND_PATH)

    # Calculate the interactions
    interactions = pc.calculate_interactions()

    # Check if the interactions are calculated correctly
    assert interactions.shape[1] == 9, "Interactions calculation is incorrect."

    # Test calculate_rmsd method
    example_ligand = load_mols_from_sdf(EXAMPLE_LIGAND_PATH)[0]

    rmsd_out = pc.calculate_rmsd(example_ligand, example_ligand)

    assert rmsd_out == 0.0, "RMSD calculation is incorrect."
