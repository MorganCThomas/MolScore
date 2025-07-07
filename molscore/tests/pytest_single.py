import os
import sys
import time
import uuid
from pathlib import Path
from typing import Union

import pytest
from rdkit.Chem import AllChem as Chem

from molscore import MockGenerator, MolScore

if len(sys.argv) == 1:
    print(
        """
Usage: pytest pytest_configs.py CONFIGS...

MolScore integration test.

Arguments:
  --configs  One or more configuration definitions, paths, or directories.

Description:
  This script runs a (very) simple MolScore integration test on provided configurations.
  It initializes a MockGenerator, runs the MolScore for each config/objective, and performs scoring.
  This is mostly to check that it runs, or which errors are thrown (some should be expected).

Examples:
  pytest pytest_configs.py GuacaMol:Albuterol_similarity path/to/my_config.json

  This will run the MolScore for Albuterol_similarity from the GuacaMol benchmark, and my_config.
            """
    )


BATCH_SIZE = 10


@pytest.fixture
def configs(request):
    return request.config.getoption("--configs")


@pytest.mark.parametrize("recalculate", [True, False])
@pytest.mark.parametrize("score_only", [True, False])
def test_v1_api(
    configs: Union[str, os.PathLike],
    recalculate: bool,
    score_only: bool,
    smiles_generator,
    test_out_dir,
    ):
    for config in configs:
        time.sleep(1)  # Otherwise same file name error
        ms = MolScore(
            model_name="test",
            task_config=config,
            output_dir=test_out_dir,
            replay_size=0,
        )
        for _ in range(5):
            smiles = smiles_generator.sample(BATCH_SIZE)
            _ = ms(smiles=smiles, recalculate=recalculate, score_only=score_only)

        ms.write_scores()
        ms.kill_monitor()
        
        # Simple test
        if not score_only:
            assert (Path(ms.save_dir) / "scores.csv").exists(), "scores.csv doesn't exist"


@pytest.mark.parametrize("recalculate", [True, False])
@pytest.mark.parametrize("canonicalise_smiles", [True, False])
@pytest.mark.parametrize("score_only", [True, False])
@pytest.mark.parametrize("replay_size", [0, 50])
@pytest.mark.parametrize("mol_id", [None, "smiles", "random"])
def test_v2_api(
    configs: Union[str, os.PathLike],
    mol_id: None,
    score_only: bool,
    replay_size: int,
    canonicalise_smiles: bool,
    recalculate: bool,
    smiles_generator,
    test_out_dir
):
    for config in configs:
        time.sleep(1)  # Otherwise same file name error
        ms = MolScore(
            model_name="test",
            task_config=config,
            output_dir=test_out_dir,
            replay_size=replay_size,
        )

        # Test score
        with ms as scoring_function:
            for _ in range(5):
                kwargs = {
                    "score_only": score_only,
                    "canonicalise_smiles": canonicalise_smiles,
                    "recalculate": recalculate,
                }
                smiles = smiles_generator.sample(BATCH_SIZE)
                kwargs["smiles"] = smiles
                if mol_id and mol_id == "smiles":
                    kwargs["mol_id"] = smiles
                if mol_id and mol_id == "random":
                    kwargs["mol_id"] = [str(uuid.uuid4()) for _ in range(BATCH_SIZE)]
                
                _ = scoring_function.score(**kwargs)
            
            # Test replay_buffer
            if replay_size and not score_only:
                mols, scores = scoring_function.replay(n=10, molecule_key='smiles')
                assert len(mols) == len(scores)
                assert len(mols) > 0

        # Simple test
        if not score_only:
            assert (Path(ms.save_dir) / "scores.csv").exists(), "scores.csv doesn't exist"
            
            
@pytest.mark.parametrize("recalculate", [True, False])
@pytest.mark.parametrize("budget", [None, 50])
@pytest.mark.parametrize("oracle_budget", [None, 50])
def test_v2_termination(
    configs: Union[str, os.PathLike],
    recalculate: bool,
    budget: int,
    oracle_budget: int,
    smiles_generator,
    test_out_dir
):
    for config in configs:
        time.sleep(1)  # Otherwise same file name error
        ms = MolScore(
            model_name="test",
            task_config=config,
            output_dir=test_out_dir,
            budget=budget,
            oracle_budget=oracle_budget,
        )

        # Test score
        molecule_count = 0
        it_count = 0
        with ms as scoring_function:
            while not scoring_function.finished:
                smiles = smiles_generator.sample(BATCH_SIZE)
                molecule_count += len(smiles)
                it_count += 1
                _ = scoring_function.score(smiles, recalculate=recalculate)
                
                # Termination tests
                if (budget and oracle_budget) or oracle_budget:
                    # Smiles_generator generates 0.2 of invalid / duplicates
                    if int(molecule_count * 0.8) >= oracle_budget:
                        assert scoring_function.finished, "Scoring function should be finished"
                    if scoring_function.finished:
                        assert molecule_count >= oracle_budget, "Molecule count should be greater due to invalid and duplicates"
                
                elif budget:
                    if molecule_count >= budget:
                        assert scoring_function.finished, "Scoring function should be finished"
                
                else:
                    # No budget, break loop manually
                    if it_count >= 5:
                        break
                    
        # Simple test
        assert (Path(ms.save_dir) / "scores.csv").exists(), "scores.csv doesn't exist"


@pytest.mark.parametrize("recalculate", [True, False])
@pytest.mark.parametrize("score_only", [True, False])
@pytest.mark.parametrize("replay_size", [0, 50])
@pytest.mark.parametrize("mol_id", [None, "smiles", "random"])
def test_v2_mols(
    configs: Union[str, os.PathLike],
    mol_id: None,
    score_only: bool,
    replay_size: int,
    recalculate: bool,
    test_out_dir,
    smiles_generator
):
    for config in configs:
        time.sleep(1)  # Otherwise same file name error
        ms = MolScore(
            model_name="test",
            task_config=config,
            output_dir=test_out_dir,
            replay_size=replay_size,
        )

        # Test score
        with ms as scoring_function:
            for _ in range(5):
                kwargs = {
                    "score_only": score_only,
                    "recalculate": recalculate,
                }
                smiles = smiles_generator.sample(BATCH_SIZE)
                kwargs["rdkit_mols"] = [
                    Chem.MolFromSmiles(smi, sanitize=False) for smi in smiles
                ]
                if mol_id and mol_id == "smiles":
                    kwargs["mol_id"] = smiles
                if mol_id and mol_id == "random":
                    kwargs["mol_id"] = [str(uuid.uuid4()) for _ in range(BATCH_SIZE)]

                try:
                    _ = scoring_function.score(**kwargs)
                except TypeError as e:
                    msg = str(e)
                    if msg == "__call__() missing 1 required positional argument: 'smiles'":
                        pytest.xfail(reason=msg)
                    else:
                        raise e
        
            # Test replay buffer
            if replay_size and not score_only:
                mols, scores = scoring_function.replay(n=10, molecule_key='rdkit_mols')
                assert len(mols) == len(scores)
                assert len(mols) > 0
                
        # Simple test
        if not score_only:
            assert (Path(ms.save_dir) / "scores.csv").exists(), "scores.csv doesn't exist"


if __name__ == "__main__":
    pytest.main([__file__])
