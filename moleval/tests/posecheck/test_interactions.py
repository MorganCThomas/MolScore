import unittest

import pandas as pd

from moleval.metrics.posecheck.utils.constants import EXAMPLE_LIGAND_PATH, EXAMPLE_PDB_PATH
from moleval.metrics.posecheck.utils.interactions import (calculate_interaction_similarity,
                                    generate_interaction_df,
                                    merge_interaction_dfs)
from moleval.metrics.posecheck.utils.loading import load_mols_from_sdf, load_protein_from_pdb


class TestInteractions(unittest.TestCase):
    def setUp(self):
        self.prot = load_protein_from_pdb(EXAMPLE_PDB_PATH)
        self.lig = load_mols_from_sdf(EXAMPLE_LIGAND_PATH)

    def test_generate_interaction_df(self):
        interaction_df = generate_interaction_df(self.prot, self.lig)
        self.assertIsInstance(interaction_df, pd.DataFrame)
        # Add more assertions as needed

    def test_merge_interaction_dfs(self):
        interaction_df = generate_interaction_df(self.prot, self.lig)
        merged_df = merge_interaction_dfs(interaction_df, interaction_df)
        self.assertIsInstance(merged_df, pd.DataFrame)
        # Add more assertions as needed

    def test_calculate_interaction_similarity(self):
        interaction_df = generate_interaction_df(self.prot, self.lig)
        merged_df = merge_interaction_dfs(interaction_df, interaction_df)
        similarities = calculate_interaction_similarity(merged_df)
        self.assertEqual(similarities, [1.0])
        # Add more assertions as needed


if __name__ == "__main__":
    unittest.main()
