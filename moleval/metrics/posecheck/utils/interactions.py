from typing import List, Optional

import pandas as pd
import prolif as plf
from rdkit import DataStructs


def generate_interaction_df(
    prot: plf.Molecule,
    lig: plf.Molecule,
    count: bool = False,
    drop_empty: bool = True,
    residues: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Generate a DataFrame with all interactions between protein and ligand.

    Args:
        prot: A protein molecule of type plf.Molecule.
        lig: A ligand molecule of type plf.Molecule.

    Returns:
        A DataFrame representing all interactions between the protein and ligand.
    """
    fp = plf.Fingerprint(count=count)
    fp.run_from_iterable(lig, prot, residues=residues)
    df = fp.to_dataframe(drop_empty=drop_empty)
    return df


def merge_interaction_dfs(ref_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge two interaction DataFrames into one.

    Args:
        ref_df: The reference interaction DataFrame of type pd.DataFrame.
        df: The interaction DataFrame to merge with the reference of type pd.DataFrame.

    Returns:
        A merged DataFrame containing interactions from both input DataFrames.
    """

    ref_df.rename({0: "ref"}, inplace=True)
    # drop the ligand level on both dataframes
    ref_df.columns = ref_df.columns.droplevel(0)
    df.columns = df.columns.droplevel(0)
    # concatenate and sort columns
    df = (
        pd.concat([ref_df, df])
        .fillna(False)
        .sort_index(
            axis=1,
            level=0,
            key=lambda index: [plf.ResidueId.from_string(x) for x in index],
        )
    )
    return df


def calculate_interaction_similarity(df: Optional[pd.DataFrame]) -> List[float]:
    """
    Calculate the Tanimoto similarity between the reference ligand and all other ligands.

    Args:
        df: A DataFrame containing interaction data. If None, an empty list is returned.

    Returns:
        A list of Tanimoto similarity values between the reference ligand and all other ligands.
    """

    if df is None:
        return []

    bvs = plf.to_bitvectors(df)
    similarities = []
    for i, bv in enumerate(bvs[1:]):
        tc = DataStructs.TanimotoSimilarity(bvs[0], bv)
        # print(f"{i}: {tc:.3f}")
        similarities.append(round(tc, 2))
    return similarities
