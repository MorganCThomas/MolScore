import unittest
import json
from molscore.tests import test_files
from molscore.tests.mock_generator import MockGenerator
from molscore.tests.base_tests import BaseTests
from molscore.scoring_functions.reaction_filter import DecoratedReactionFilter

class TestDecoratedReactionFilter(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Instantiate
        cls.obj = DecoratedReactionFilter
        cls.inst = DecoratedReactionFilter(
            prefix='test',
            scaffold='C1CCn(CC1)CC',
            libinvent_reactions=True,
            n_jobs=1,
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = [
            'Cc1nc2n(c(=O)c1CCN1CCC(c3noc4cc(F)ccc34)CC1)CCCC2',
            'O=c1[nH]c2ccccc2n1CCCN1CCC(n2c(=O)[nH]c3cc(Cl)ccc32)CC1',
            'CCCCCCCCCCCCCCCC(=O)OC1CCCn2c1nc(C)c(CCN1CCC(c3noc4cc(F)ccc34)CC1)c2=O',
            'Cc1nc2ccccn2c(=O)c1CCN1CCc2oc3ccccc3c2C1',
            'O=c1c(CO)coc2cc(OCCCN3CCC(c4noc5cc(F)ccc45)CC3)ccc12'
            ]
        cls.output = cls.inst(smiles=cls.input)
        print(f"\nDecoratedReactionFilter Output:\n{json.dumps(cls.output, indent=2)}\n")


class TestDecoratedReactionFilterParallel(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Instantiate
        cls.obj = DecoratedReactionFilter
        cls.inst = DecoratedReactionFilter(
            prefix='test',
            scaffold='C1CCn(CC1)CC',
            libinvent_reactions=True,
            n_jobs=4,
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = [
            'Cc1nc2n(c(=O)c1CCN1CCC(c3noc4cc(F)ccc34)CC1)CCCC2',
            'O=c1[nH]c2ccccc2n1CCCN1CCC(n2c(=O)[nH]c3cc(Cl)ccc32)CC1',
            'CCCCCCCCCCCCCCCC(=O)OC1CCCn2c1nc(C)c(CCN1CCC(c3noc4cc(F)ccc34)CC1)c2=O',
            'Cc1nc2ccccn2c(=O)c1CCN1CCc2oc3ccccc3c2C1',
            'O=c1c(CO)coc2cc(OCCCN3CCC(c4noc5cc(F)ccc45)CC3)ccc12'
            ]
        cls.output = cls.inst(smiles=cls.input)
        print(f"\nDecoratedReactionFilterParallel Output:\n{json.dumps(cls.output, indent=2)}\n")

if __name__ == '__main__':
    unittest.main()
