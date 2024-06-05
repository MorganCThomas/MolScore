"""This file needs to be run with PyTest and not Unittest"""

import json
import unittest
from random import sample

from molscore.scoring_functions.admet_ai import ADMETAI
from molscore.scoring_functions.chemprop import ChemPropModel
from molscore.scoring_functions.legacy_qsar import LegacyQSAR
from molscore.scoring_functions.molskill import MolSkill
from molscore.scoring_functions.pidgin import PIDGIN
from molscore.scoring_functions.rascore_xgb import RAScore_XGB
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


class TestLegacyQSAR(BaseTests.TestScoringFunction):
    @classmethod
    def setUpClass(cls):
        # Instantiate
        cls.obj = LegacyQSAR
        cls.inst = LegacyQSAR(
            prefix="DRD2",
            env_engine="mamba",
            model="molopt_DRD2",
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = mg.sample(5)
        cls.output = cls.inst(smiles=cls.input)
        print(f"LegacyQSAR Model Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        cls.inst._kill_server()


class TestRAScore(BaseTests.TestScoringFunction):
    @classmethod
    def setUpClass(cls):
        # Instantiate
        cls.obj = RAScore_XGB
        cls.inst = RAScore_XGB(
            prefix="RAScore",
            env_engine="mamba",
            model="ChEMBL",
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = mg.sample(5)
        cls.output = cls.inst(smiles=cls.input)
        print(f"RAScore Model Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        cls.inst._kill_server()


class TestPIDGIN(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
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
            thresh=sample(["100 uM", "10 uM", "1 uM"], 1)[0],
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
        cls.output = cls.inst(smiles=cls.input)
        print(f"\nPIDGIN Output:\n{json.dumps(cls.output, indent=2)}\n")


class TestMolSkill(BaseTests.TestScoringFunction):
    @classmethod
    def setUpClass(cls):
        # Instantiate
        cls.obj = MolSkill
        cls.inst = MolSkill(
            prefix="MolSkill",
            env_engine="mamba",
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = mg.sample(5)
        cls.output = cls.inst(smiles=cls.input)
        print(f"MolSkill Model Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        cls.inst._kill_server()


if __name__ == "__main__":
    unittest.main()
