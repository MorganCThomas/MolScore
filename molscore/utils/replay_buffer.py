import logging
import warnings
from typing import Union

import pandas as pd

from molscore.utils.chem_utils import augment_smiles, canonicalize_smiles

class ReplayBuffer:
    def __init__(self, size: int, purge: bool = True):
        self.size = size
        self.purge = purge
        self.buffer = pd.DataFrame()

    def load(self, file_path: str):
        self.buffer = pd.read_csv(
            file_path,
            index_col=0,
            dtype={"Unnamed: 0": "int64", "valid": object, "unique": object},
        )

    def save(self, file_path: str):
        self.buffer.to_csv(file_path)

    def update(self, df: pd.DataFrame, endpoint: str, using_DF: bool = False):
        df = df.copy()
        # Purge df
        if self.purge:
            if using_DF:
                df = df.loc[df.passes_diversity_filter, :]
        # Concat
        self.buffer = pd.concat([self.buffer, df], axis=0)
        # Drop_duplicates
        self.buffer.drop_duplicates(subset="smiles", inplace=True)
        # Sort
        self.buffer.sort_values(by=endpoint, ascending=False, inplace=True)
        # Prune
        self.buffer = self.buffer.iloc[: self.size, :]

    def sample(self, n, endpoint: str, augment: bool = False) -> Union[list, list]:
        """
        Sample n molecules from the replay buffer
        :param n: Number of molecules to sample
        :param augment: Whether to augment the replay buffer by randomizing the smiles
        :return: List of SMILES and scores
        """
        if n > len(self.buffer):
            n = len(self.buffer)
        # Sample n from buffer
        sample_df = self.buffer.sample(n=n)
        smiles = sample_df.smiles.tolist()
        scores = sample_df[endpoint].tolist()
        # Augment
        if augment:
            smiles = augment_smiles(smiles)
        return smiles, scores

    def reset(self):
        self.buffer = pd.DataFrame()

    def __len__(self):
        return len(self.buffer)

    def __contains__(self, smiles):
        can_smiles = canonicalize_smiles(smiles)
        return can_smiles in self.buffer.smiles.values
