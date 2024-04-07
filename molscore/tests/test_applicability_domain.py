import json
import unittest

from molscore.scoring_functions import ApplicabilityDomain
from molscore.tests import BaseTests, MockGenerator


class TestAD(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Clean the output directory
        mg = MockGenerator(seed_no=0)
        ref_smiles = mg.sample(500)
        # Instantiate
        cls.obj = ApplicabilityDomain
        cls.inst = ApplicabilityDomain(
            prefix="test", ref_smiles=ref_smiles, fp="ECFP4c", qed=True, physchem=True
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = mg.sample(5)
        cls.output = cls.inst(smiles=cls.input)
        print(f"\nAD Output:\n{json.dumps(cls.output, indent=2)}\n")


class TestADParallel(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Clean the output directory
        mg = MockGenerator(seed_no=0)
        ref_smiles = mg.sample(500)
        # Instantiate
        cls.obj = ApplicabilityDomain
        cls.inst = ApplicabilityDomain(
            prefix="test",
            ref_smiles=ref_smiles,
            fp="ECFP4c",
            qed=True,
            physchem=True,
            n_jobs=4,
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = mg.sample(5)
        cls.output = cls.inst(smiles=cls.input)
        print(f"\nAD Output:\n{json.dumps(cls.output, indent=2)}\n")


if __name__ == "__main__":
    unittest.main()
