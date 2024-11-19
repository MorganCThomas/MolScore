import unittest

import datamol as dm

from moleval.metrics.posecheck import PoseCheck
from moleval.metrics.posecheck.utils.constants import EXAMPLE_LIGAND_PATH, EXAMPLE_PDB_PATH


class TestPoseCheck(unittest.TestCase):
    def setUp(self):
        self.pc = PoseCheck()
        self.pc.load_protein_from_pdb(EXAMPLE_PDB_PATH)
        self.pc.load_ligands_from_sdf(EXAMPLE_LIGAND_PATH)

    def test_load_ligands_from_rdmol(self):
        mol = dm.read_sdf(EXAMPLE_LIGAND_PATH)[0]
        pc_test = PoseCheck()
        pc_test.load_ligands_from_mols(mol)

    def test_calculate_clashes(self):
        clashes = self.pc.calculate_clashes()[0]
        self.assertEqual(clashes, 2, "Clashes calculation is incorrect.")

    def test_calculate_strain_energy(self):
        strain_energy = self.pc.calculate_strain_energy()[0]
        self.assertAlmostEqual(strain_energy, 19.11, places=2)

    def test_calculate_interactions(self):
        interactions = self.pc.calculate_interactions()
        self.assertEqual(
            interactions.shape[1], 9, "Interactions calculation is incorrect."
        )

    def test_calculate_rmsd(self):
        example_ligand = dm.read_sdf(EXAMPLE_LIGAND_PATH)[0]
        rmsd_out = self.pc.calculate_rmsd(example_ligand, example_ligand)
        self.assertEqual(rmsd_out, 0.0, "RMSD calculation is incorrect.")

if __name__ == "__main__":
    unittest.main()
