import json
import os
import unittest

from molscore.scoring_functions.aizynthfinder import AiZynthFinder
from molscore.scoring_functions.rascore_xgb import RAScore_XGB
from molscore.tests import BaseTests, MockGenerator


class TestAiZynthFinder(BaseTests.TestScoringFunction):
    @classmethod
    def setUpClass(cls):
        # Clean the output directory
        os.makedirs(cls.output_directory, exist_ok=True)
        # Instantiate
        cls.obj = AiZynthFinder
        cls.inst = AiZynthFinder(n_jobs=5)
        # Call
        mg = MockGenerator(seed_no=123, augment_invalids=True)
        cls.input = mg.sample(10)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(
            smiles=cls.input, directory=cls.output_directory, file_names=file_names
        )
        print(f"\AiZynthFinder Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        os.system(f"rm -r {os.path.join(cls.output_directory, '*')}")


class TestRAScore(BaseTests.TestScoringFunction):
    @classmethod
    def setUpClass(cls):
        # Clean the output directory
        os.makedirs(cls.output_directory, exist_ok=True)
        # Instantiate
        cls.obj = RAScore_XGB
        cls.inst = RAScore_XGB(n_jobs=5)
        # Call
        mg = MockGenerator(seed_no=123, augment_invalids=True)
        cls.input = mg.sample(10)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(
            smiles=cls.input, directory=cls.output_directory, file_names=file_names
        )
        print(f"\RAScore Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        os.system(f"rm -r {os.path.join(cls.output_directory, '*')}")


if __name__ == "__main__":
    unittest.main()
