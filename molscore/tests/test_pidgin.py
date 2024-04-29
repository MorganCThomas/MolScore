import json
import os
import unittest
from random import sample

from molscore.scoring_functions import PIDGIN
from molscore.tests import BaseTests, MockGenerator


class TestPIDGIN(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Clean the output directory
        os.makedirs(cls.output_directory, exist_ok=True)
        # Instantiate
        cls.obj = PIDGIN
        PIDGIN.set_docstring()
        uniprots = sample(PIDGIN.get_uniprot_list(), 10)
        groups = sample(list(PIDGIN.get_uniprot_groups().keys()), 1)
        cls.inst = PIDGIN(
            prefix="test",
            uniprot=uniprots[0],
            uniprots=uniprots[1:],
            uniprot_set=groups[0],
            thresh=sample(["100 uM", "10 uM", "1 uM", "0.1 uM"], 1)[0],
            n_jobs=1,
            method=sample(["mean", "median", "max", "mean"], 1)[0],
            binarise=sample([False, False], 1)[0],
        )
        print(
            f"\nPIDGIN Input: {len(cls.inst.uniprots)} uniprots, {cls.inst.thresh} threshold, method={cls.inst.method}, binarise={cls.inst.binarise}\n"
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = mg.sample(5)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(
            smiles=cls.input, directory=cls.output_directory, file_names=file_names
        )
        print(f"\nPIDGIN Output:\n{json.dumps(cls.output, indent=2)}\n")


class TestPIDGINParallel(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Clean the output directory
        os.makedirs(cls.output_directory, exist_ok=True)
        # Instantiate
        cls.obj = PIDGIN
        PIDGIN.set_docstring()
        uniprots = sample(PIDGIN.get_uniprot_list(), 10)
        groups = sample(list(PIDGIN.get_uniprot_groups().keys()), 1)
        cls.inst = PIDGIN(
            prefix="test",
            uniprot=uniprots[0],
            uniprots=uniprots[1:],
            uniprot_set=groups[0],
            thresh=sample(["100 uM", "10 uM", "1 uM", "0.1 uM"], 1)[0],
            n_jobs=4,
            method=sample(["mean", "median", "max", "mean"], 1)[0],
            binarise=sample([False, False], 1)[0],
        )
        print(
            f"\nPIDGIN Input: {len(cls.inst.uniprots)} uniprots, {cls.inst.thresh} threshold, method={cls.inst.method}, binarise={cls.inst.binarise}\n"
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = mg.sample(5)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(
            smiles=cls.input, directory=cls.output_directory, file_names=file_names
        )
        print(f"\nPIDGIN Output:\n{json.dumps(cls.output, indent=2)}\n")


if __name__ == "__main__":
    unittest.main()
