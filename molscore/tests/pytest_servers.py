"""This file needs to be run with PyTest and not Unittest"""

import json
import unittest

from molscore.scoring_functions.admet_ai import ADMETAI
from molscore.scoring_functions.chemprop import ChemPropModel
from molscore.scoring_functions.legacy_qsar import LegacyQSAR
from molscore.tests import BaseTests, MockGenerator, test_files

"""Note this fails due to Flask not handling unittesting correctly https://stackoverflow.com/questions/46792087/flask-unit-test-failed-to-establish-a-new-connection"""


class TestLibINVENT(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Instantiate
        cls.obj = LegacyQSAR
        cls.inst = LegacyQSAR(
            prefix="DRD2",
            env_engine="mamba",
            model="libinvent_DRD2",
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = mg.sample(5)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(
            smiles=cls.input, directory=cls.output_directory, file_names=file_names
        )
        print(f"LibINVENT Model Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        cls.inst._kill_server()


class TestADMETAI(BaseTests.TestScoringFunction):
    @classmethod
    def setUpClass(cls):
        # Instantiate
        cls.obj = ADMETAI
        cls.inst = ADMETAI(
            prefix="ADMETAI",
            env_engine="mamba",
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = mg.sample(5)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(
            smiles=cls.input, directory=cls.output_directory, file_names=file_names
        )
        print(f"ADMET-AI Model Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        cls.inst._kill_server()


class TestChemProp(BaseTests.TestScoringFunction):
    @classmethod
    def setUpClass(cls):
        # Instantiate
        cls.obj = ChemPropModel
        cls.inst = ChemPropModel(
            prefix="test",
            env_engine="mamba",
            model_dir=test_files["chemprop_model"],
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = mg.sample(5)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(
            smiles=cls.input, directory=cls.output_directory, file_names=file_names
        )
        print(f"ChemProp Model Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        cls.inst._kill_server()


if __name__ == "__main__":
    unittest.main()
