import os
import json
import unittest
from random import sample

from molscore.tests import test_files
from molscore.tests.base_tests import BaseTests
from molscore.tests.mock_generator import MockGenerator
from molscore.scoring_functions.legacy_qsar import LegacyQSAR

"""Note this fails due to Flask not handling unittesting correctly https://stackoverflow.com/questions/46792087/flask-unit-test-failed-to-establish-a-new-connection"""
class TestLibINVENT(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Instantiate
        cls.obj = LegacyQSAR
        cls.inst = LegacyQSAR(
            prefix='DRD2',
            env_engine='mamba',
            model='libinvent_DRD2',
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = mg.sample(5)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(smiles=cls.input, directory=cls.output_directory, file_names=file_names)
        print(f"\LibINVENT Model Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        cls.inst._kill_server()

if __name__ == '__main__':
    unittest.main()