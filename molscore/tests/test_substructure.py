import json
import unittest

from molscore.scoring_functions.substructure_match import SubstructureMatch
from molscore.scoring_functions.substructure_filters import SubstructureFilters
from molscore.tests import BaseTests, MockGenerator


class TestSubstructureMatch(BaseTests.TestScoringFunction):
    @classmethod
    def setUpClass(cls):
        mg = MockGenerator(seed_no=123)
        cls.obj = SubstructureMatch
        cls.inst = SubstructureMatch(
            prefix="test",
            smarts=["[#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]-c12"],
            n_jobs=1,
            method="any",
        )
        cls.input = mg.sample(5)
        cls.output = cls.inst(smiles=cls.input)
        print(f"SubstructureMatch Output:\n{json.dumps(cls.output, indent=2)}\n")


class TestSubstructureFilters(BaseTests.TestScoringFunction):
    @classmethod
    def setUpClass(cls):
        mg = MockGenerator(seed_no=123)
        cls.obj = SubstructureFilters
        cls.inst = SubstructureFilters(
            prefix="test",
            az_filters=True,
            mcf_filters=True,
            pains_filters=True,
            n_jobs=1,
            method="any",
        )
        cls.input = mg.sample(5)
        cls.output = cls.inst(smiles=cls.input)
        print(f"SubstructureFilters Output:\n{json.dumps(cls.output, indent=2)}\n")


if __name__ == "__main__":
    unittest.main()