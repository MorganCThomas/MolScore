import json
import unittest

from molscore.scoring_functions.similarity import (
    MolecularSimilarity,
    LevenshteinSimilarity,
    TanimotoSimilarity,
)
from molscore.tests import BaseTests, MockGenerator


class TestTanimotoECFP4MeanSingle(BaseTests.TestScoringFunction):
    def setUp(self):
        mg = MockGenerator(seed_no=123)
        self.obj = TanimotoSimilarity
        self.inst = TanimotoSimilarity(
            prefix="test",
            ref_smiles=mg.sample(10),
            radius=2,
            bits=1024,
            features=False,
            counts=False,
            method="mean",
            n_jobs=1,
        )
        self.input = mg.sample(64)
        self.output = self.inst(self.input)


class TestTanimotoECFP4MaxSingle(BaseTests.TestScoringFunction):
    def setUp(self):
        mg = MockGenerator(seed_no=123)
        self.obj = TanimotoSimilarity
        self.inst = TanimotoSimilarity(
            prefix="test",
            ref_smiles=mg.sample(10),
            radius=2,
            bits=1024,
            features=False,
            counts=False,
            method="max",
            n_jobs=1,
        )
        self.input = mg.sample(64)
        self.output = self.inst(self.input)


class TestTanimotoFCFP4MeanSingle(BaseTests.TestScoringFunction):
    def setUp(self):
        mg = MockGenerator(seed_no=123)
        self.obj = TanimotoSimilarity
        self.inst = TanimotoSimilarity(
            prefix="test",
            ref_smiles=mg.sample(10),
            radius=2,
            bits=1024,
            features=True,
            counts=False,
            method="mean",
            n_jobs=1,
        )
        self.input = mg.sample(64)
        self.output = self.inst(self.input)


class TestTanimotoFCFP4MaxSingle(BaseTests.TestScoringFunction):
    def setUp(self):
        mg = MockGenerator(seed_no=123)
        self.obj = TanimotoSimilarity
        self.inst = TanimotoSimilarity(
            prefix="test",
            ref_smiles=mg.sample(10),
            radius=2,
            bits=1024,
            features=True,
            counts=False,
            method="max",
            n_jobs=1,
        )
        self.input = mg.sample(64)
        self.output = self.inst(self.input)


class TestTanimotoECFC4MeanSingle(BaseTests.TestScoringFunction):
    def setUp(self):
        mg = MockGenerator(seed_no=123)
        self.obj = TanimotoSimilarity
        self.inst = TanimotoSimilarity(
            prefix="test",
            ref_smiles=mg.sample(10),
            radius=2,
            bits=1024,
            features=False,
            counts=True,
            method="mean",
            n_jobs=1,
        )
        self.input = mg.sample(64)
        self.output = self.inst(self.input)


class TestTanimotoECFC4MaxSingle(BaseTests.TestScoringFunction):
    def setUp(self):
        mg = MockGenerator(seed_no=123)
        self.obj = TanimotoSimilarity
        self.inst = TanimotoSimilarity(
            prefix="test",
            ref_smiles=mg.sample(10),
            radius=2,
            bits=1024,
            features=False,
            counts=True,
            method="max",
            n_jobs=1,
        )
        self.input = mg.sample(64)
        self.output = self.inst(self.input)


class TestTanimotoFCFC4MeanSingle(BaseTests.TestScoringFunction):
    def setUp(self):
        mg = MockGenerator(seed_no=123)
        self.obj = TanimotoSimilarity
        self.inst = TanimotoSimilarity(
            prefix="test",
            ref_smiles=mg.sample(10),
            radius=2,
            bits=1024,
            features=True,
            counts=True,
            method="mean",
            n_jobs=1,
        )
        self.input = mg.sample(64)
        self.output = self.inst(self.input)


class TestTanimotoFCFC4MaxSingle(BaseTests.TestScoringFunction):
    def setUp(self):
        mg = MockGenerator(seed_no=123)
        self.obj = TanimotoSimilarity
        self.inst = TanimotoSimilarity(
            prefix="test",
            ref_smiles=mg.sample(10),
            radius=2,
            bits=1024,
            features=True,
            counts=True,
            method="max",
            n_jobs=1,
        )
        self.input = mg.sample(64)
        self.output = self.inst(self.input)


class TestTanimotoFCFC4MeanParallel(BaseTests.TestScoringFunction):
    def setUp(self):
        mg = MockGenerator(seed_no=123)
        self.obj = TanimotoSimilarity
        self.inst = TanimotoSimilarity(
            prefix="test",
            ref_smiles=mg.sample(10),
            radius=2,
            bits=1024,
            features=True,
            counts=True,
            method="mean",
            n_jobs=6,
        )
        self.input = mg.sample(64)
        self.output = self.inst(self.input)


class TestTanimotoFCFC4MaxParallel(BaseTests.TestScoringFunction):
    def setUp(self):
        mg = MockGenerator(seed_no=123)
        self.obj = TanimotoSimilarity
        self.inst = TanimotoSimilarity(
            prefix="test",
            ref_smiles=mg.sample(10),
            radius=2,
            bits=1024,
            features=True,
            counts=True,
            method="max",
            n_jobs=6,
        )
        self.input = mg.sample(64)
        self.output = self.inst(self.input)


# ---- Test new version ----


class TestSimilarityECFP4(BaseTests.TestScoringFunction):
    @classmethod
    def setUpClass(cls):
        mg = MockGenerator(seed_no=123)
        # Instantiate
        cls.obj = MolecularSimilarity
        cls.inst = MolecularSimilarity(
            prefix="test",
            ref_smiles=mg.sample(10),
            fp="ECFP4",
            thresh=None,
            method="max",
            n_jobs=1,
        )
        print("\nSimilarity Input: fp=ECFP4, method=max")
        # Call
        cls.input = mg.sample(5)
        cls.output = cls.inst(smiles=cls.input)
        print(f"Similarity Output:\n{json.dumps(cls.output, indent=2)}\n")


class TestSimilarityECFP4Parallel(BaseTests.TestScoringFunction):
    @classmethod
    def setUpClass(cls):
        mg = MockGenerator(seed_no=123)
        # Instantiate
        cls.obj = MolecularSimilarity
        cls.inst = MolecularSimilarity(
            prefix="test",
            ref_smiles=mg.sample(10),
            fp="ECFP4",
            thresh=None,
            method="max",
            n_jobs=4,
        )
        print("\nSimilarity Input: fp=ECFP4, method=max")
        # Call
        cls.input = mg.sample(5)
        cls.output = cls.inst(smiles=cls.input)
        print(f"Similarity Output:\n{json.dumps(cls.output, indent=2)}\n")


class TestSimilarityECFP4ThreshMeanParallel(BaseTests.TestScoringFunction):
    @classmethod
    def setUpClass(cls):
        mg = MockGenerator(seed_no=123)
        # Instantiate
        cls.obj = MolecularSimilarity
        cls.inst = MolecularSimilarity(
            prefix="test",
            ref_smiles=mg.sample(100),
            fp="ECFP4",
            thresh=0.35,
            method="mean",
            n_jobs=4,
        )
        print("\nSimilarity Input: fp=ECFP4, thresh=0.35, method=mean")
        # Call
        cls.input = mg.sample(5)
        cls.output = cls.inst(smiles=cls.input)
        print(f"Similarity Output:\n{json.dumps(cls.output, indent=2)}\n")


class TestSimilarityECFP4ThreshMaxParallel(BaseTests.TestScoringFunction):
    @classmethod
    def setUpClass(cls):
        mg = MockGenerator(seed_no=123)
        # Instantiate
        cls.obj = MolecularSimilarity
        cls.inst = MolecularSimilarity(
            prefix="test",
            ref_smiles=mg.sample(100),
            fp="ECFP4",
            thresh=0.35,
            method="max",
            n_jobs=4,
        )
        print("\nSimilarity Input: fp=ECFP4, thresh=0.35, method=max")
        # Call
        cls.input = mg.sample(5)
        cls.output = cls.inst(smiles=cls.input)
        print(f"Similarity Output:\n{json.dumps(cls.output, indent=2)}\n")


# ---- Test Levenshtein Similarity ----


class TestLevenshteinSimilarity(BaseTests.TestScoringFunction):
    @classmethod
    def setUpClass(cls):
        mg = MockGenerator(seed_no=123)
        # Instantiate
        cls.obj = LevenshteinSimilarity
        cls.inst = LevenshteinSimilarity(
            prefix="test",
            ref_smiles=mg.sample(10),
            n_jobs=1,
        )
        print("\nLevenshtein Similarity Input")
        # Call
        cls.input = mg.sample(5)
        cls.output = cls.inst(smiles=cls.input)
        print(f"Levenshtein Similarity Output:\n{json.dumps(cls.output, indent=2)}\n")



if __name__ == "__main__":
    unittest.main()
