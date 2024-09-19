import unittest

import datamol as dm

from moleval.metrics.posecheck.utils.constants import EXAMPLE_LIGAND_PATH, EXAMPLE_PDB_PATH
from moleval.metrics.posecheck.utils.loading import (get_ids_to_pockets, load_mols_from_rdkit,
                               load_mols_from_sdf, load_protein_from_pdb)


class LoadingTest(unittest.TestCase):
    def test_load_protein_from_pdb(self):
        prot = load_protein_from_pdb(EXAMPLE_PDB_PATH)
        self.assertIsNotNone(prot)

    def test_load_mols_from_sdf(self):
        lig = load_mols_from_sdf(EXAMPLE_LIGAND_PATH)[0]
        self.assertIsNotNone(lig)

    def test_load_mols_from_rdkit(self):
        mol = dm.read_sdf(EXAMPLE_LIGAND_PATH)[0]
        lig = load_mols_from_rdkit(mol)
        self.assertIsNotNone(lig)
        self.assertIsInstance(lig, list)

    #def test_get_ids_to_pockets(self):
    #    names = get_ids_to_pockets()
    #    self.assertIsNotNone(names)


if __name__ == "__main__":
    unittest.main()
