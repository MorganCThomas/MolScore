import unittest

from moleval.metrics.posecheck.utils.clashes import count_clashes, count_clashes_list
from moleval.metrics.posecheck.utils.constants import EXAMPLE_LIGAND_PATH, EXAMPLE_PDB_PATH
from moleval.metrics.posecheck.utils.loading import load_mols_from_sdf, load_protein_from_pdb


class ClashesTest(unittest.TestCase):
    def setUp(self):
        # Define a test protein and ligand
        self.prot = load_protein_from_pdb(EXAMPLE_PDB_PATH)
        self.lig = load_mols_from_sdf(EXAMPLE_LIGAND_PATH)[0]

    def test_count_clashes(self):
        # Count clashes between protein and ligand
        clashes = count_clashes(self.prot, self.lig)

        # Assert that the number of clashes is correct
        self.assertEqual(clashes, 2)

    def test_count_clashes_list(self):
        # Count clashes between protein and ligands
        clashes_list = count_clashes_list(self.prot, [self.lig, self.lig])

        # Assert that the number of clashes for each ligand is correct
        self.assertEqual(clashes_list[0]["clashes"], 2)
        self.assertEqual(clashes_list[1]["clashes"], 2)


if __name__ == "__main__":
    unittest.main()
