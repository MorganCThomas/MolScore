import unittest
from unittest.mock import patch
from rdkit import Chem
from openbabel import pybel
import numpy as np
from molscore.scoring_functions.hsr_sim import HSR
from molscore.tests import test_files
import os
from hsr import pre_processing as pp
from hsr.utils import PROTON_FEATURES

class TestHSR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # cls.reference_molecule_file = test_files["DRD2_ref_ligand"]  # Reference molecule file path
        cwd = os.getcwd()
        cls.reference_molecule_file = f"{cwd}/data/HSR_test_ref_molecule.sdf"
        cls.test_smiles = [
            "Clc1cc(O)c(Cl)cc1C(=O)c1ccccc1",
            "Clc1cc(O)c(Cl)cc1c1ccccc1",
            "Clc1cc(O)c(Cl)cc1C(=O)"
        ]
        cls.mols_3d = [f'{cwd}/data/HSR_mol_{i}.sdf' for i in range(1, 4)]
        cls.test_rdkit_molecules = [pp.read_mol_from_file(mol) for mol in cls.mols_3d]
        cls.test_pybel_molecules = [next(pybel.readfile("sdf", mol)) for mol in cls.mols_3d]
        cls.test_np_arrays = []
        for mol in cls.test_rdkit_molecules:
            # Generate random 3D coordinates
            cls.test_np_arrays.append(pp.molecule_to_ndarray(mol, features=PROTON_FEATURES))
            
        try:
            from ccdc.molecule import Molecule as ccdcMolecule
            cls.test_ccdc_molecules = [
                ccdcMolecule.from_string(smi) for smi in cls.test_smiles
            ]
        except Exception as e:
            cls.test_ccdc_molecules = []
            print("CCDC module not available. Skipping CCDC-related tests.")

    def test_hsr_with_smiles(self):
        for generator in ["rdkit", "obabel",]: # "ccdc"]:
            with self.subTest(generator=generator):
                inst = HSR(
                    prefix=f"test_{generator}",
                    ref_molecule=self.reference_molecule_file,
                    generator=generator,
                )
                results = inst.score(self.test_smiles, directory="./", file_names=["1", "2", "3"])
                self.assertEqual(len(results), len(self.test_smiles))
                if generator == "rdkit":
                    self.assertEqual(round(results[0][f"test_{generator}_HSR_score"], 2), 0.87,)
                    self.assertEqual(round(results[1][f"test_{generator}_HSR_score"],2), 0.89,) 
                    self.assertEqual(round(results[2][f"test_{generator}_HSR_score"],2), 0.65,)
                elif generator == "obabel":
                    self.assertEqual(round(results[0][f"test_{generator}_HSR_score"],2), 0.91,)
                    self.assertEqual(round(results[1][f"test_{generator}_HSR_score"],2), 0.89,) 
                    self.assertEqual(round(results[2][f"test_{generator}_HSR_score"],2), 0.67,)
                

    def test_hsr_with_rdkit_molecules(self):
        inst = HSR(
            prefix="test_rdkit",
            ref_molecule=self.reference_molecule_file,
            generator="rdkit",
        )
        results = inst.score(self.test_rdkit_molecules, directory="./", file_names=["1", "2", "3"])
        self.assertEqual(len(results), len(self.test_rdkit_molecules))
        self.assertEqual(round(results[0]["test_rdkit_HSR_score"],2), 1.00,)
        self.assertEqual(round(results[1]["test_rdkit_HSR_score"],2), 0.89,)
        self.assertEqual(round(results[2]["test_rdkit_HSR_score"],2), 0.66,)

    def test_hsr_with_pybel_molecules(self):
        inst = HSR(
            prefix="test_pybel",
            ref_molecule=self.reference_molecule_file,
            generator="obabel",
        )
        results = inst.score(self.test_pybel_molecules, directory="./", file_names=["1", "2", "3"])
        self.assertEqual(len(results), len(self.test_pybel_molecules))
        self.assertEqual(round(results[0]["test_pybel_HSR_score"],2), 1.00,)
        self.assertEqual(round(results[1]["test_pybel_HSR_score"],2), 0.89,)
        self.assertEqual(round(results[2]["test_pybel_HSR_score"],2), 0.66,)

    def test_hsr_with_ccdc_molecules(self):
        #TODO: Update this test
        if not self.test_ccdc_molecules:
            self.skipTest("CCDC module not available. Skipping CCDC molecule tests.")
        inst = HSR(
            prefix="test_ccdc",
            ref_molecule=self.reference_molecule_file,
            generator="ccdc",
        )
        results = inst.score(self.test_ccdc_molecules, directory="./", file_names=["1", "2", "3"])
        self.assertEqual(len(results), len(self.test_ccdc_molecules))
        for result in results:
            self.assertGreaterEqual(result["test_ccdc_HSR_score"], 0.0)

    def test_hsr_with_np_arrays(self):
        inst = HSR(
            prefix="test_numpy",
            ref_molecule=self.reference_molecule_file,
            generator='None', 
        )
        results = inst.score(self.test_np_arrays, directory="./", file_names=["1", "2", "3"])
        self.assertEqual(len(results), len(self.test_np_arrays))
        self.assertEqual(round(results[0]["test_numpy_HSR_score"],2), 1.00,)
        self.assertEqual(round(results[1]["test_numpy_HSR_score"],2), 0.89,)
        self.assertEqual(round(results[2]["test_numpy_HSR_score"],2), 0.66,)


if __name__ == "__main__":
    unittest.main()
