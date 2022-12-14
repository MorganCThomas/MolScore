import os
import unittest
import subprocess

from molscore.tests import test_files
from molscore.tests.base_tests import BaseTests
from molscore.tests.mock_generator import MockGenerator
from molscore.scoring_functions import GlideDock, SminaDock


class TestGlideDockSerial(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):

        # Check installation
        if not ('SCHRODINGER' in list(os.environ.keys())):
            raise unittest.SkipTest("Schrodinger installation not found")
        # Check license
        license_check = subprocess.run("$SCHRODINGER/licadmin STAT", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode().split("\n")
        for line in license_check:
            if 'Error getting status:' in line:
                raise unittest.SkipTest(line)

        # Clean the output directory
        os.makedirs(cls.output_directory, exist_ok=True)
        # Prepare a grid file
        input_file = os.path.join(cls.output_directory, 'glide.in')
        with open(input_file, 'w') as f:
            f.write(f"GRIDFILE   {test_files['GlideDock_grid']}\n")
            f.write(f"PRECISION    SP\n")
        # Instantiate
        cls.obj = GlideDock
        cls.inst = GlideDock(
            prefix='test',
            glide_template=input_file,
            ligand_preparation='LigPrep',
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = mg.sample(5)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(smiles=cls.input, directory=cls.output_directory, file_names=file_names)
        print(f"\nGlideDock Output:\n{cls.output}\n")

    @classmethod
    def tearDownClass(cls):
        os.system(f"rm -r {os.path.join(cls.output_directory, '*')}")


class TestGlideDockParallel(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):

        # Check installation
        if not ('SCHRODINGER' in list(os.environ.keys())):
            raise unittest.SkipTest("Schrodinger installation not found")
        # Check license
        license_check = subprocess.run("$SCHRODINGER/licadmin STAT", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode().split("\n")
        for line in license_check:
            if 'Error getting status:' in line:
                raise unittest.SkipTest(line)

        # Clean the output directory
        os.makedirs(cls.output_directory, exist_ok=True)
        # Prepare a grid file
        input_file = os.path.join(cls.output_directory, 'glide.in')
        with open(input_file, 'w') as f:
            f.write(f"GRIDFILE   {test_files['GlideDock_grid']}\n")
            f.write(f"PRECISION    SP\n")
        # Instantiate
        cls.obj = GlideDock
        cls.inst = GlideDock(
            prefix='test',
            glide_template=input_file,
            ligand_preparation='LigPrep',
            cluster=5
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = mg.sample(5)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(smiles=cls.input, directory=cls.output_directory, file_names=file_names)
        print(f"\nGlideDock Output:\n{cls.output}\n")

    @classmethod
    def tearDownClass(cls):
        os.system(f"rm -r {os.path.join(cls.output_directory, '*')}")


class TestSminaDock(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Clean output directory
        os.makedirs(cls.output_directory, exist_ok=True)
        os.system(f"rm -r {os.path.join(cls.output_directory, '*')}")
        # Instantiate
        cls.obj = SminaDock
        cls.inst = SminaDock(
            prefix='test',
            receptor=test_files['SminaDock_receptor'],
            ref_ligand=test_files['SminaDock_ref_ligand'],
            cpus=8,
            ligand_preparation='GypsumDL',
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = mg.sample(5)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(smiles=cls.input, directory=cls.output_directory, file_names=file_names)
        print(f"\nSminaDock Output:\n{cls.output}\n")

    @classmethod
    def tearDownClass(cls):
        os.system(f"rm -r {os.path.join(cls.output_directory, '*')}")


if __name__ == '__main__':
    unittest.main()