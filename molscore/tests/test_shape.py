import json
import os
import unittest

from molscore.scoring_functions import Align3D
from molscore.tests import BaseTests, MockGenerator, test_files


class TestAlign3D(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Clean the output directory
        os.makedirs(cls.output_directory, exist_ok=True)
        # Instantiate
        cls.obj = Align3D
        cls.inst = Align3D(
            prefix="test",
            ref_smiles=["Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"],
            ref_sdf=test_files["DRD2_ref_ligand"],
            similarity_method="Tanimoto",
            agg_method="mean",
            pharmacophore_similarity=True,
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = mg.sample(5)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(
            smiles=cls.input, directory=cls.output_directory, file_names=file_names
        )
        print(f"\nAlign3D Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        os.system(f"rm -r {os.path.join(cls.output_directory, '*')}")


class TestAlign3DParallel(BaseTests.TestScoringFunction):
    # Only set up once per class, otherwise too long
    @classmethod
    def setUpClass(cls):
        # Clean the output directory
        os.makedirs(cls.output_directory, exist_ok=True)
        # Instantiate
        cls.obj = Align3D
        cls.inst = Align3D(
            prefix="test",
            ref_smiles=["Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"],
            ref_sdf=test_files["DRD2_ref_ligand"],
            similarity_method="Tanimoto",
            agg_method="mean",
            pharmacophore_similarity=True,
            n_jobs=5,
        )
        # Call
        mg = MockGenerator(seed_no=123)
        cls.input = mg.sample(5)
        file_names = [str(i) for i in range(len(cls.input))]
        cls.output = cls.inst(
            smiles=cls.input, directory=cls.output_directory, file_names=file_names
        )
        print(f"\nAlign3D Output:\n{json.dumps(cls.output, indent=2)}\n")

    @classmethod
    def tearDownClass(cls):
        os.system(f"rm -r {os.path.join(cls.output_directory, '*')}")


if __name__ == "__main__":
    unittest.main()
