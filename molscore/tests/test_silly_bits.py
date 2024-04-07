import json
import tempfile
import unittest

from molscore.scoring_functions.silly_bits import SillyBits
from molscore.scoring_functions.utils import write_smiles
from molscore.tests import BaseTests, MockGenerator


class TestSillyBits(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Instantiate
        cls.obj = SillyBits
        # Create a temporary file for the reference smiles
        mg = MockGenerator(seed_no=123)
        with tempfile.NamedTemporaryFile(mode="wt") as f:
            ref_sample = mg.sample(1000)
            write_smiles(ref_sample, f.name)
            cls.inst = SillyBits(
                prefix="test",
                reference_smiles=f.name,
                n_jobs=1,
            )
            # Call
            cls.input = mg.sample(5)
            cls.output = cls.inst(smiles=cls.input)
            print(f"\n{cls.__name__} Output:\n{json.dumps(cls.output, indent=2)}\n")


class TestSillyBitsParallel(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Instantiate
        cls.obj = SillyBits
        # Create a temporary file for the reference smiles
        mg = MockGenerator(seed_no=123)
        with tempfile.NamedTemporaryFile(mode="wt") as f:
            ref_sample = mg.sample(1000)
            write_smiles(ref_sample, f.name)
            cls.inst = SillyBits(
                prefix="test",
                reference_smiles=f.name,
                n_jobs=4,
            )
            # Call
            cls.input = mg.sample(5)
            cls.output = cls.inst(smiles=cls.input)
            print(f"\n{cls.__name__} Output:\n{json.dumps(cls.output, indent=2)}\n")


if __name__ == "__main__":
    unittest.main()
