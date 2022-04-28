import unittest
from molscore.test.mock_generator import MockGenerator
from molscore.test.tests.base_tests import BaseTests
from molscore.scoring_functions.similarity import TanimotoSimilarity


class TestTanimotoECFP4MeanSingle(BaseTests.TestScoringFunction):
    def setUp(self):
        mg = MockGenerator(seed_no=123)
        self.cls = TanimotoSimilarity
        self.inst = TanimotoSimilarity(
            prefix='test',
            ref_smiles=mg.sample(10),
            radius=2,
            bits=1024,
            features=False,
            counts=False,
            method='mean',
            n_jobs=1
        )
        self.input = mg.sample(64)
        self.output = self.inst(self.input)


class TestTanimotoECFP4MaxSingle(BaseTests.TestScoringFunction):
    def setUp(self):
        mg = MockGenerator(seed_no=123)
        self.cls = TanimotoSimilarity
        self.inst = TanimotoSimilarity(
            prefix='test',
            ref_smiles=mg.sample(10),
            radius=2,
            bits=1024,
            features=False,
            counts=False,
            method='max',
            n_jobs=1
        )
        self.input = mg.sample(64)
        self.output = self.inst(self.input)


class TestTanimotoFCFP4MeanSingle(BaseTests.TestScoringFunction):
    def setUp(self):
        mg = MockGenerator(seed_no=123)
        self.cls = TanimotoSimilarity
        self.inst = TanimotoSimilarity(
            prefix='test',
            ref_smiles=mg.sample(10),
            radius=2,
            bits=1024,
            features=True,
            counts=False,
            method='mean',
            n_jobs=1
        )
        self.input = mg.sample(64)
        self.output = self.inst(self.input)


class TestTanimotoFCFP4MaxSingle(BaseTests.TestScoringFunction):
    def setUp(self):
        mg = MockGenerator(seed_no=123)
        self.cls = TanimotoSimilarity
        self.inst = TanimotoSimilarity(
            prefix='test',
            ref_smiles=mg.sample(10),
            radius=2,
            bits=1024,
            features=True,
            counts=False,
            method='max',
            n_jobs=1
        )
        self.input = mg.sample(64)
        self.output = self.inst(self.input)


class TestTanimotoECFC4MeanSingle(BaseTests.TestScoringFunction):
    def setUp(self):
        mg = MockGenerator(seed_no=123)
        self.cls = TanimotoSimilarity
        self.inst = TanimotoSimilarity(
            prefix='test',
            ref_smiles=mg.sample(10),
            radius=2,
            bits=1024,
            features=False,
            counts=True,
            method='mean',
            n_jobs=1
        )
        self.input = mg.sample(64)
        self.output = self.inst(self.input)


class TestTanimotoECFC4MaxSingle(BaseTests.TestScoringFunction):
    def setUp(self):
        mg = MockGenerator(seed_no=123)
        self.cls = TanimotoSimilarity
        self.inst = TanimotoSimilarity(
            prefix='test',
            ref_smiles=mg.sample(10),
            radius=2,
            bits=1024,
            features=False,
            counts=True,
            method='max',
            n_jobs=1
        )
        self.input = mg.sample(64)
        self.output = self.inst(self.input)


class TestTanimotoFCFC4MeanSingle(BaseTests.TestScoringFunction):
    def setUp(self):
        mg = MockGenerator(seed_no=123)
        self.cls = TanimotoSimilarity
        self.inst = TanimotoSimilarity(
            prefix='test',
            ref_smiles=mg.sample(10),
            radius=2,
            bits=1024,
            features=True,
            counts=True,
            method='mean',
            n_jobs=1
        )
        self.input = mg.sample(64)
        self.output = self.inst(self.input)


class TestTanimotoFCFC4MaxSingle(BaseTests.TestScoringFunction):
    def setUp(self):
        mg = MockGenerator(seed_no=123)
        self.cls = TanimotoSimilarity
        self.inst = TanimotoSimilarity(
            prefix='test',
            ref_smiles=mg.sample(10),
            radius=2,
            bits=1024,
            features=True,
            counts=True,
            method='max',
            n_jobs=1
        )
        self.input = mg.sample(64)
        self.output = self.inst(self.input)


class TestTanimotoFCFC4MeanParallel(BaseTests.TestScoringFunction):
    def setUp(self):
        mg = MockGenerator(seed_no=123)
        self.cls = TanimotoSimilarity
        self.inst = TanimotoSimilarity(
            prefix='test',
            ref_smiles=mg.sample(10),
            radius=2,
            bits=1024,
            features=True,
            counts=True,
            method='mean',
            n_jobs=6
        )
        self.input = mg.sample(64)
        self.output = self.inst(self.input)


class TestTanimotoFCFC4MaxParallel(BaseTests.TestScoringFunction):
    def setUp(self):
        mg = MockGenerator(seed_no=123)
        self.cls = TanimotoSimilarity
        self.inst = TanimotoSimilarity(
            prefix='test',
            ref_smiles=mg.sample(10),
            radius=2,
            bits=1024,
            features=True,
            counts=True,
            method='max',
            n_jobs=6
        )
        self.input = mg.sample(64)
        self.output = self.inst(self.input)


if __name__ == '__main__':
    unittest.main()
