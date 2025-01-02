import os
import sys
import time
import uuid
from typing import Union

import pytest
from rdkit.Chem import AllChem as Chem

from molscore import MockGenerator, MolScore

if len(sys.argv) == 1:
    print(
        """
Usage: python pytest_configs.py CONFIGS...

MolScore integration test.

Arguments:
  --configs  One or more configuration definitions, paths, or directories.

Description:
  This script runs a (very) simple MolScore integration test on provided configurations (think objectives).
  It initializes a MockGenerator, runs the MolScore for each config/objective, and performs scoring.
  This is mostly to check that it runs, or which errors are thrown (some should be expected).

Examples:
  python pytest_configs.py GuacaMol:Albuterol_similarity path/to/my_config.json

  This will run the MolScore for Albuterol_similarity from the GuacaMol benchmark, my_config, and all JSON configuration files in my_configs.
            """
    )


MG = MockGenerator(augment_invalids=True, augment_duplicates=True)
BATCH_SIZE = 10


@pytest.fixture
def configs(request):
    return request.config.getoption("--configs")


@pytest.mark.parametrize("recalculate", [True, False])
@pytest.mark.parametrize("score_only", [True, False])
def test_v1(configs: Union[str, os.PathLike], recalculate: bool, score_only: bool):
    for config in configs:
        time.sleep(1)  # Otherwise same file name error
        ms = MolScore(
            model_name="test",
            task_config=config,
            output_dir=os.path.join(os.path.dirname(__file__), "test_out"),
            replay_size=0,
        )

        for _ in range(5):
            smiles = MG.sample(BATCH_SIZE)
            _ = ms(smiles=smiles, recalculate=recalculate, score_only=score_only)

        ms.write_scores()
        ms.kill_monitor()


@pytest.mark.parametrize("recalculate", [True, False])
@pytest.mark.parametrize("canonicalise_smiles", [True, False])
@pytest.mark.parametrize("score_only", [True, False])
@pytest.mark.parametrize("replay_size", [0, 50])
@pytest.mark.parametrize("mol_id", [None, "smiles", "random"])
def test_v2_smiles(
    configs: Union[str, os.PathLike],
    mol_id: None,
    score_only: bool,
    replay_size: int,
    canonicalise_smiles: bool,
    recalculate: bool,
):
    for config in configs:
        time.sleep(1)  # Otherwise same file name error
        ms = MolScore(
            model_name="test",
            task_config=config,
            output_dir=os.path.join(os.path.dirname(__file__), "test_out"),
            replay_size=replay_size,
        )

        # Test score
        for _ in range(5):
            kwargs = {
                "score_only": score_only,
                "canonicalise_smiles": canonicalise_smiles,
                "recalculate": recalculate,
            }
            smiles = MG.sample(BATCH_SIZE)
            kwargs["smiles"] = smiles
            if mol_id and mol_id == "smiles":
                kwargs["mol_id"] = smiles
            if mol_id and mol_id == "random":
                kwargs["mol_id"] = [str(uuid.uuid4()) for _ in range(BATCH_SIZE)]

            _ = ms.score(**kwargs)
            
        # Test replay_buffer
        if replay_size and not score_only:
            mols, scores = ms.replay(n=10, molecule_key='smiles')
            assert len(mols) == len(scores)
            assert len(mols) > 0

        ms.write_scores()
        ms.kill_monitor()


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
):
    for config in configs:
        time.sleep(1)  # Otherwise same file name error
        ms = MolScore(
            model_name="test",
            task_config=config,
            output_dir=os.path.join(os.path.dirname(__file__), "test_out"),
            replay_size=replay_size,
        )

        # Test score
        for _ in range(5):
            kwargs = {
                "score_only": score_only,
                "recalculate": recalculate,
            }
            smiles = MG.sample(BATCH_SIZE)
            kwargs["rdkit_mols"] = [
                Chem.MolFromSmiles(smi, sanitize=False) for smi in smiles
            ]
            if mol_id and mol_id == "smiles":
                kwargs["mol_id"] = smiles
            if mol_id and mol_id == "random":
                kwargs["mol_id"] = [str(uuid.uuid4()) for _ in range(BATCH_SIZE)]

            try:
                _ = ms.score(**kwargs)
            except TypeError as e:
                msg = str(e)
                if msg == "__call__() missing 1 required positional argument: 'smiles'":
                    pytest.xfail(reason=msg)
                else:
                    raise e
        
        # Test replay buffer
        if replay_size and not score_only:
            mols, scores = ms.replay(n=10, molecule_key='rdkit_mols')
            assert len(mols) == len(scores)
            assert len(mols) > 0

        ms.write_scores()
        ms.kill_monitor()


if __name__ == "__main__":
    pytest.main([__file__])
