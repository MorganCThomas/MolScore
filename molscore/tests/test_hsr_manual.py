import unittest
from unittest.mock import patch
from rdkit import Chem
from openbabel import pybel
import numpy as np
from molscore.scoring_functions.hsr import HSR
from molscore.tests import test_files
import os
from hsr import pre_processing as pp
from hsr.utils import PROTON_FEATURES


cwd = os.getcwd()
reference_molecule_file = f"{cwd}/data/HSR_test_ref_molecule.sdf"
test_smiles = [
    "Clc1cc(O)c(Cl)cc1C(=O)c1ccccc1",
    "Clc1cc(O)c(Cl)cc1c1ccccc1",
    "Clc1cc(O)c(Cl)cc1C(=O)"
]

mols_3d = [f'{cwd}/data/HSR_mol_{i}.sdf' for i in range(1, 4)]

test_rdkit_molecules = [Chem.MolFromSmiles(smi) for smi in test_smiles]

test_rdkit_molecules_3d = [pp.read_mol_from_file(mol) for mol in mols_3d]
# Generate 3D coordinates for RDKit molecules
# test_rdkit_molecules_3d = []
# for mol in test_rdkit_molecules:
#     # Generate 3D coordinates
#     mol = Chem.AddHs(mol)
#     Chem.AllChem.EmbedMolecule(mol)
#     Chem.AllChem.MMFFOptimizeMolecule(mol)
#     test_rdkit_molecules_3d.append(mol)
    
test_pybel_molecules = [pybel.readstring("smi", smi) for smi in test_smiles]

test_pybel_molecules_3d = [next(pybel.readfile("sdf", mol)) for mol in mols_3d]
# test_pybel_molecules_3d = []
# for mol in test_pybel_molecules:
#     mol.addh()
#     mol.make3D()
#     mol.localopt()
#     test_pybel_molecules_3d.append(mol)
    
test_np_arrays = []
for mol in test_rdkit_molecules_3d:
    # Generate random 3D coordinates
    test_np_arrays.append(pp.molecule_to_ndarray(mol, features=PROTON_FEATURES))

def test_hsr_with_smiles():
    for generator in ["rdkit", "obabel",]: #  "ccdc"]:
            inst = HSR(
                prefix=f"test_{generator}",
                ref_molecule=reference_molecule_file,
                generator=generator,
                save_files=True,
            )
            results = inst.score(test_smiles, directory="./", file_names=["1", "2", "3"])
            # self.assertEqual(len(results), len(self.test_smiles))
            print(f'SMILES with {generator}')
            for i,result in enumerate(results):
                # proint only the score
                score = result[f"test_{generator}_HSR_score"]
                print(f'{i}- {score:.2f}')   
                # self.assertGreaterEqual(result[f"test_{generator}_HSR_score"], 0.0)

def test_hsr_with_rdkit_molecules():
    inst = HSR(
        prefix="test_rdkit",
        ref_molecule=reference_molecule_file,
        generator="rdkit",
        save_files=True,
    )
    results = inst.score(test_rdkit_molecules_3d, directory="./", file_names=["1", "2", "3"])
    # self.assertEqual(len(results), len(self.test_rdkit_molecules))
    print('RDKit Molecules')
    for i, result in enumerate(results):
        score = result['test_rdkit_HSR_score']
        print(f'{i}- {score:.2f}')
        # self.assertGreaterEqual(result["test_rdkit_HSR_score"], 0.0)

def test_hsr_with_pybel_molecules():
    inst = HSR(
        prefix="test_pybel",
        ref_molecule=reference_molecule_file,
        generator="obabel",
        save_files=True,
    )
    results = inst.score(test_pybel_molecules_3d, directory="./", file_names=["1", "2", "3"])
    # self.assertEqual(len(results), len(self.test_pybel_molecules))
    print('Pybel Molecules')
    for i, result in enumerate(results):
        score = result['test_pybel_HSR_score']
        print(f'{i}- {score:.2f}')
        # self.assertGreaterEqual(result["test_pybel_HSR_score"], 0.0)

def test_hsr_with_np_arrays():
    inst = HSR(
        prefix="test_numpy",
        ref_molecule=reference_molecule_file,
        generator=None,  
        save_files=True,
    )
    results = inst.score(test_np_arrays, directory="./", file_names=["1", "2", "3"])
    # self.assertEqual(len(results), len(self.test_np_arrays))
    print('Numpy Arrays')
    for i, result in enumerate(results):
        score = result['test_numpy_HSR_score']
        print(f'{i}- {score:.2f}')
        # self.assertGreaterEqual(result["test_numpy_HSR_score"], 0.0)
        
# def test_hsr_with_ccdc_molecules(self):
#     if not self.test_ccdc_molecules:
#         self.skipTest("CCDC module not available. Skipping CCDC molecule tests.")
#     inst = HSR(
#         prefix="test_ccdc",
#         ref_molecule=self.reference_molecule_file,
#         generator="ccdc",
#     )
#     results = inst.score(self.test_ccdc_molecules, directory="./", file_names=["1", "2", "3"])
#     self.assertEqual(len(results), len(self.test_ccdc_molecules))
#     for result in results:
#         self.assertGreaterEqual(result["test_ccdc_HSR_score"], 0.0)

if __name__ == "__main__":
    test_hsr_with_smiles()
    test_hsr_with_rdkit_molecules()
    test_hsr_with_pybel_molecules()
    test_hsr_with_np_arrays()

