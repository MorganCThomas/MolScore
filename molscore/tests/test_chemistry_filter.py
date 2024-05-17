import json
import unittest

from molscore.tests import BaseTests, MockGenerator
from molscore.scoring_functions.chemistry_filters import ChemistryFilter


class TestCF(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Clean the output directory
        mg = MockGenerator(seed_no=0)
        ref_smiles = mg.sample(500)
        # Instantiate
        cls.obj = ChemistryFilter
        cls.inst = ChemistryFilter(
            prefix="test", ref_smiles=ref_smiles
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = mg.sample(5)
        cls.output = cls.inst(smiles=cls.input)
        print(f"\nCF Output:\n{json.dumps(cls.output, indent=2)}\n")


if __name__ == "__main__":
    unittest.main()
