import json
import unittest

from molscore.scoring_functions.reaction_filter import (
    DecoratedReactionFilter,
    SelectiveDecoratedReactionFilter,
)
from molscore.tests import BaseTests


class TestDecoratedReactionFilter(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Instantiate
        cls.obj = DecoratedReactionFilter
        cls.inst = DecoratedReactionFilter(
            prefix="test",
            scaffold="C1CCN(CC1)CC",
            libinvent_reactions=True,
            n_jobs=1,
        )
        # Call
        cls.input = [
            "Cc1nc2n(c(=O)c1CCN1CCC(c3noc4cc(F)ccc34)CC1)CCCC2",
            "O=c1[nH]c2ccccc2n1CCCN1CCC(n2c(=O)[nH]c3cc(Cl)ccc32)CC1",
            "CCCCCCCCCCCCCCCC(=O)OC1CCCn2c1nc(C)c(CCN1CCC(c3noc4cc(F)ccc34)CC1)c2=O",
            "Cc1nc2ccccn2c(=O)c1CCN1CCc2oc3ccccc3c2C1",
            "O=c1c(CO)coc2cc(OCCCN3CCC(c4noc5cc(F)ccc45)CC3)ccc12",
        ]
        cls.output = cls.inst(smiles=cls.input)
        print(
            f"\nDecoratedReactionFilter Output:\n{json.dumps(cls.output, indent=2)}\n"
        )


class TestDecoratedReactionFilterParallel(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Instantiate
        cls.obj = DecoratedReactionFilter
        cls.inst = DecoratedReactionFilter(
            prefix="test",
            scaffold="C1CCN(CC1)CC",
            libinvent_reactions=True,
            n_jobs=4,
        )
        # Call
        cls.input = [
            "Cc1nc2n(c(=O)c1CCN1CCC(c3noc4cc(F)ccc34)CC1)CCCC2",
            "O=c1[nH]c2ccccc2n1CCCN1CCC(n2c(=O)[nH]c3cc(Cl)ccc32)CC1",
            "CCCCCCCCCCCCCCCC(=O)OC1CCCn2c1nc(C)c(CCN1CCC(c3noc4cc(F)ccc34)CC1)c2=O",
            "Cc1nc2ccccn2c(=O)c1CCN1CCc2oc3ccccc3c2C1",
            "O=c1c(CO)coc2cc(OCCCN3CCC(c4noc5cc(F)ccc45)CC3)ccc12",
        ]
        cls.output = cls.inst(smiles=cls.input)
        print(
            f"\nDecoratedReactionFilterParallel Output:\n{json.dumps(cls.output, indent=2)}\n"
        )


class TestSelectiveDecoratedReactionFilter(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Instantiate
        cls.obj = SelectiveDecoratedReactionFilter
        cls.inst = SelectiveDecoratedReactionFilter(
            prefix="test",
            scaffold="[N:1]1CCN(CC1)CCCC[N:0]",
            allowed_reactions={
                0: [
                    "[#6;!$(C(C=*)(C=*));!$([#6]~[O,N,S]);$([#6]~[#6]):1][C:2](=[O:3])[N;D2;$(N(C=[O,S]));!$(N~[O,P,S,N]):4][#6;!$(C=*);!$([#6](~[O,N,S])N);$([#6]~[#6]):5]>>[#6:1][C:2](=[O:3])[*].[*][N:4][#6:5]"
                ],
                1: [
                    "[c;$(c1:[c,n]:[c,n]:[c,n]:[c,n]:[c,n]:1):1]-!@[N;$(NC)&!$(N=*)&!$([N-])&!$(N#*)&!$([ND1])&!$(N[O])&!$(N[C,S]=[S,O,N]),H2&$(Nc1:[c,n]:[c,n]:[c,n]:[c,n]:[c,n]:1):2]>>[*][c;$(c1:[c,n]:[c,n]:[c,n]:[c,n]:[c,n]:1):1].[*][N:2]"
                ],
            },
            n_jobs=1,
        )
        # Call
        cls.input = [
            "O=C1c2ccccc2S(=O)(=O)N1CCCCN1CCN(c2ncccn2)CC1",
            "O=C1CC2(CCCC2)CC(=O)N1CCCCN1CCN(c2ncccn2)CC1",
            "COc1ccccc1N1CCN(CCCCN2C(=O)c3cccc4cccc2c34)CC1",
            "CC(=O)NCCCCN1CCN(c2ccc(Cl)cc2)CC1",
            "CC1(C(=O)NCCCCN2CCN(c3ccc(Cl)cc3)CC2)CCCCC1",
        ]
        cls.output = cls.inst(smiles=cls.input)
        print(
            f"\nSelectiveDecoratedReactionFilter Output:\n{json.dumps(cls.output, indent=2)}\n"
        )


if __name__ == "__main__":
    unittest.main()
