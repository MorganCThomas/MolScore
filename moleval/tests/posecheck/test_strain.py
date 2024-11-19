import unittest

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from moleval.metrics.posecheck.utils.strain import calculate_energy, get_strain_energy, relax_mol


class EnergyCalculationTest(unittest.TestCase):
    def test_calculate_energy(self):
        # Define a test molecule

        # smile for large molecule
        smiles = "CC(CC1=CC=C(C=C1)C(=O)O)C1=CC=C(C=C1)C(=O)O"

        mol = Chem.MolFromSmiles(smiles)
        # add generate conformer using UFF
        AllChem.EmbedMolecule(mol, randomSeed=0xF00D)

        # Calculate the energy using the UFF force field
        energy = calculate_energy(mol, forcefield="UFF", add_hs=True)

        # Assert that the calculated energy is not NaN
        self.assertFalse(np.isnan(energy))


class MoleculeRelaxationTest(unittest.TestCase):
    def test_relax_mol(self):
        # Define a test molecule
        smiles = "CCO"
        mol = Chem.MolFromSmiles(smiles)

        # Relax the molecule
        relaxed_mol = relax_mol(mol)

        # Assert that the relaxed molecule is not None
        self.assertIsNotNone(relaxed_mol)

        # Assert that the relaxed molecule has coordinates
        self.assertTrue(relaxed_mol.GetNumConformers() > 0)


class StrainEnergyCalculationTest(unittest.TestCase):
    def test_get_strain_energy(self):
        # Define a test molecule

        # smile for large molecule
        smiles = "CC(CC1=CC=C(C=C1)C(=O)O)C1=CC=C(C=C1)C(=O)O"

        mol = Chem.MolFromSmiles(smiles)
        # add generate conformer using UFF
        AllChem.EmbedMolecule(mol, randomSeed=0xF00D)

        # Calculate the strain energy
        strain_energy = get_strain_energy(mol)

        # Assert that the calculated strain energy is not NaN
        self.assertFalse(np.isnan(strain_energy))


if __name__ == "__main__":
    unittest.main()
