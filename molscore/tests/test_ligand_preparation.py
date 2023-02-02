import unittest
import logging
import os
import subprocess

from molscore.tests.base_tests import BaseTests
from molscore.tests.mock_generator import MockGenerator
from molscore.scoring_functions._ligand_preparation import LigPrep, Epik, Moka, GypsumDL


class TestLigPrep(BaseTests.TestLigandPreparation):
    def setUp(self):
            # Check installation
            if not ('SCHRODINGER' in list(os.environ.keys())):
                raise unittest.SkipTest('Schrodinger installation not found')
            # Check license
            license_check = subprocess.run("$SCHRODINGER/licadmin STAT", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode().split("\n")
            for line in license_check:
                if 'Error getting status:' in line:
                    raise unittest.SkipTest(line)
            mg = MockGenerator(seed_no=123)
            self.obj = LigPrep
            logger = logging.getLogger('test')
            logger.setLevel(logging.DEBUG)
            self.inst = LigPrep(
                logger=logger
            )
            # Clean the output directory
            os.makedirs(self.output_directory, exist_ok=True)
            self.input = mg.sample(5)
            file_names = [str(i) for i in range(len(self.input))]
            self.output = self.inst(smiles=self.input, directory=self.output_directory, file_names=file_names)

    def tearDown(self):
        os.system(f"rm -r {os.path.join(self.output_directory, '*')}")

class TestEpik(BaseTests.TestLigandPreparation):
    def setUp(self):
            # Check installation
            if not 'SCHRODINGER' in list(os.environ.keys()):
                    raise unittest.SkipTest('Schrodinger installation not found')
            # Check license
            license_check = subprocess.run("$SCHRODINGER/licadmin STAT", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode().split("\n")
            for line in license_check:
                if 'Error getting status:' in line:
                    raise unittest.SkipTest(line)
            mg = MockGenerator(seed_no=123)
            self.obj = Epik
            logger = logging.getLogger('test')
            logger.setLevel(logging.DEBUG)
            self.inst = Epik(
                logger=logger
            )
            # Clean the output directory
            os.makedirs(self.output_directory, exist_ok=True)
            self.input = mg.sample(5)
            file_names = [str(i) for i in range(len(self.input))]
            self.output = self.inst(smiles=self.input, directory=self.output_directory, file_names=file_names)

    def tearDown(self):
        os.system(f"rm -r {os.path.join(self.output_directory, '*')}")

class TestMoka(BaseTests.TestLigandPreparation):
    def setUp(self):
            # Check installation
            moka_env = subprocess.run(args=['which', 'blabber_sd'], stdout=subprocess.PIPE).stdout.decode().strip('\n')
            corina_env = subprocess.run(args=['which', 'corina'], stdout=subprocess.PIPE).stdout.decode().strip('\n')
            if (moka_env == '') or (corina_env == ''):
                raise unittest.SkipTest('Moka or Corina installation not found')
            mg = MockGenerator(seed_no=123)
            self.obj = Moka
            logger = logging.getLogger('test')
            logger.setLevel(logging.DEBUG)
            self.inst = Moka(
                logger=logger
            )
            # Clean the output directory
            os.makedirs(self.output_directory, exist_ok=True)
            self.input = mg.sample(5)
            file_names = [str(i) for i in range(len(self.input))]
            self.output = self.inst(smiles=self.input, directory=self.output_directory, file_names=file_names)

    def tearDown(self):
        os.system(f"rm -r {os.path.join(self.output_directory, '*')}")

class TestGypsumDL(BaseTests.TestLigandPreparation):
    def setUp(self):
            mg = MockGenerator(seed_no=123)
            self.obj = GypsumDL
            logger = logging.getLogger('test')
            logger.setLevel(logging.DEBUG)
            self.inst = GypsumDL(
                logger=logger
            )
            # Clean the output directory
            os.makedirs(self.output_directory, exist_ok=True)
            self.input = mg.sample(5)
            file_names = [str(i) for i in range(len(self.input))]
            self.output = self.inst(smiles=self.input, directory=self.output_directory, file_names=file_names)

    def tearDown(self):
        os.system(f"rm -r {os.path.join(self.output_directory, '*')}")

if __name__ == '__main__':
    unittest.main()