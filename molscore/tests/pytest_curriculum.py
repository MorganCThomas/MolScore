import os
import sys

import pytest

from molscore import MolScoreCurriculum

if len(sys.argv) == 1:
    print(
        """
Usage: pytest pytest_curriculum.py [--configs CONFIG_PATH] [-v]

MolScore curriculum learning integration test.

Arguments:
  --configs CONFIG_PATH  Path to a custom curriculum directory to test.

Description:
  This script runs tests for MolScore in curriculum mode. It 
  requires a specific custom config directorycan either test.

Examples:
  pytest pytest_curriculum.py --configs path/to/my_curriculum_dir  # Test a custom curriculum
            """
    )
    
BATCH_SIZE = 10

@pytest.fixture
def configs(request):
    return request.config.getoption("--configs", default=None)

@pytest.mark.parametrize("budget", [None, 100])
@pytest.mark.parametrize("termination_threshold", [None, 0.5])
@pytest.mark.parametrize("termination_patience", [None, 5])
@pytest.mark.parametrize("termination_early_stop", [True, False])
@pytest.mark.parametrize("oracle_budget", [True, False])
def test_custom_benchmark(
    configs,
    smiles_generator,
    test_out_dir,
    budget,
    termination_threshold,
    termination_patience,
    termination_early_stop,
    oracle_budget,
    ):
    """Test a custom curriculum directory."""
    if configs is None:
        pytest.skip("No custom curriculum directory provided")
    
    for config in configs:
        if not os.path.isdir(config):
            pytest.skip(f"Custom curriculum directory '{config}' does not exist")
            
        try:
            msc = MolScoreCurriculum(
                model_name="test",
                output_dir=test_out_dir,
                budget=budget,
                termination_threshold=termination_threshold,
                termination_patience=termination_patience,
                termination_early_stop=termination_early_stop,
                oracle_budget=oracle_budget,
                custom_benchmark=config,
                run_name="Curriculum",
            )
        except Exception as e:
            pytest.fail(f"Failed to create curriculum from directory '{config}': {str(e)}")
            
        with msc as scoring_function:
            while not scoring_function.finished:
                smiles = smiles_generator.sample(BATCH_SIZE)
                scores = scoring_function.score(smiles)
                assert scores is not None, f"Scoring failed for a config in {config}"


if __name__ == "__main__":
    pytest.main([__file__])