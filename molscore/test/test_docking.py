import os
import unittest

from molscore.test import test_files
from molscore.test.base_tests import BaseTests
from molscore.test.mock_generator import MockGenerator
from molscore.scoring_functions import GlideDock, SminaDock


class TestGlideDock(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Check if we skip it
        if not ('SCHRODINGER' in list(os.environ.keys())):
            cls.skipTest('Schrodinger installation not found')
        mg = MockGenerator(seed_no=123)
        # Clean the output directory
        os.makedirs(cls.output_directory, exist_ok=True)
        # Prepare a grid file
        input_file = os.path.join(cls.output_directory, 'glide.in')
        with open(input_file, 'w') as f:
            f.write(f"GRIDFILE   {test_files['glide_grid']}\n")
            f.write(f"PRECISION    SP\n")
        # Instantiate
        cls.obj = GlideDock
        cls.inst = GlideDock(
            prefix='test',
            glide_template=input_file,
            ligand_preparation='LigPrep',
        )
        cls.input = mg.sample(5)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(smiles=cls.input, directory=cls.output_directory, file_names=file_names)

    @classmethod
    def tearDownClass(cls):
        os.system(f"rm -r {os.path.join(cls.output_directory, '*')}")

if __name__ == '__main__':
    unittest.main()