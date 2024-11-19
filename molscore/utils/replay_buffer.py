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
        )

    def save(self, file_path: str):
        self.buffer.to_csv(file_path)

    def update(self, df: pd.DataFrame, endpoint_key: str, using_DF: bool = False, **molecular_inputs):
        df = df.copy()
        # Concatenate non-existing molecular inputs
        exclude_keys = [k for k in molecular_inputs.keys() if k in df.columns]
        for k in exclude_keys: molecular_inputs.pop(k)
        df = pd.concat([df, pd.DataFrame(molecular_inputs)], axis=1)
        # Purge df
        if self.purge:
            if using_DF:
                df = df.loc[df.passes_diversity_filter, :]
        # Concat
        self.buffer = pd.concat([self.buffer, df], axis=0)
        # Drop_duplicates by mol_id
        self.buffer.drop_duplicates(subset="mol_id", inplace=True)
        # Sort
        self.buffer.sort_values(by=endpoint_key, ascending=False, inplace=True)
        # Prune
        self.buffer = self.buffer.iloc[: self.size, :]

    def sample(self, n, endpoint_key: str, molecule_key: str = 'smiles', augment: bool = False) -> Union[list, list]:
        """
        Sample n molecules from the replay buffer
        :param n: Number of molecules to sample
        :param endpoint: Column name of the endpoint e.g., 'single', 'score' etc.
        :param molecule_key: Column name of the molecular input to return e.g., 'smiles', 'rdkit_mol' etc.
        :param augment: Whether to augment the replay buffer by randomizing the smiles
        :return: List of molecular input and scores
        """
        if len(self.buffer) == 0:
            return [], []
        if n > len(self.buffer):
            n = len(self.buffer)
        # Sample n from buffer
        sample_df = self.buffer.sample(n=n)
        # Get molecular input
        mols = sample_df[molecule_key].tolist()
        # Augment
        if molecule_key == 'smiles' and augment:
            mols = augment_smiles(mols)
        scores = sample_df[endpoint_key].tolist()
        return mols, scores

    def reset(self):
        self.buffer = pd.DataFrame()

    def __len__(self):
        return len(self.buffer)

    def __contains__(self, smiles):
        can_smiles = canonicalize_smiles(smiles)
        return can_smiles in self.buffer.smiles.values
