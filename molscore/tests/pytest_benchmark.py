import os
import sys

import pytest

from molscore import MolScoreBenchmark
from molscore.scoring_functions.utils import check_exe

if len(sys.argv) == 1:
    print(
        """
Usage: pytest pytest_benchmark.py [--configs CONFIG_PATH] [-v]

MolScore benchmark integration test.

Arguments:
  --configs CONFIG_PATH  Path to a custom benchmark directory to test.
                         If not provided, all preset benchmarks will be tested.

Description:
  This script runs tests for MolScore in benchmark mode. It can either test
  all preset benchmarks or a specific custom benchmark directory.
  Each preset benchmark has its own dedicated test, making it suitable for
  generating GitHub status badges for each benchmark.

Examples:
  pytest pytest_benchmark.py  # Test all preset benchmarks
  pytest pytest_benchmark.py --configs path/to/my_benchmark_dir  # Test a custom benchmark
  pytest pytest_benchmark.py::test_benchmark_GuacaMol  # Test only GuacaMol preset
            """
    )


BATCH_SIZE = 10


@pytest.fixture
def configs(request):
    return request.config.getoption("--configs", default=None)


# Individual test functions for each preset benchmark without repetitive parameters
def test_benchmark_GuacaMol(smiles_generator, test_out_dir):
    """Test the GuacaMol benchmark."""
    _test_preset_benchmark("GuacaMol", smiles_generator, test_out_dir)


def test_benchmark_GuacaMol_Scaffold(smiles_generator, test_out_dir):
    """Test the GuacaMol_Scaffold benchmark."""
    _test_preset_benchmark("GuacaMol_Scaffold", smiles_generator, test_out_dir)


def test_benchmark_MolOpt(smiles_generator, test_out_dir):
    """Test the MolOpt benchmark."""
    _test_preset_benchmark("MolOpt", smiles_generator, test_out_dir)


def test_benchmark_MolOpt_CF(smiles_generator, test_out_dir):
    """Test the MolOpt-CF benchmark."""
    _test_preset_benchmark("MolOpt-CF", smiles_generator, test_out_dir)


def test_benchmark_MolOpt_DF(smiles_generator, test_out_dir):
    """Test the MolOpt-DF benchmark."""
    _test_preset_benchmark("MolOpt-DF", smiles_generator, test_out_dir)


def test_benchmark_5HT2A_PhysChem(smiles_generator, test_out_dir):
    """Test the 5HT2A_PhysChem benchmark."""
    _test_preset_benchmark("5HT2A_PhysChem", smiles_generator, test_out_dir)


def test_benchmark_5HT2A_Selectivity(smiles_generator, test_out_dir):
    """Test the 5HT2A_Selectivity benchmark."""
    _test_preset_benchmark("5HT2A_Selectivity", smiles_generator, test_out_dir)


@pytest.mark.skipif(not check_exe("rbdock"), reason="rDock software not found")
def test_benchmark_5HT2A_Docking(smiles_generator, test_out_dir):
    """Test the 5HT2A_Docking benchmark."""
    _test_preset_benchmark("5HT2A_Docking", smiles_generator, test_out_dir)


def test_benchmark_LibINVENT_Exp1(smiles_generator, test_out_dir):
    """Test the LibINVENT_Exp1 benchmark."""
    _test_preset_benchmark("LibINVENT_Exp1", smiles_generator, test_out_dir)


def test_benchmark_LinkINVENT_Exp3(smiles_generator, test_out_dir):
    """Test the LinkINVENT_Exp3 benchmark."""
    _test_preset_benchmark("LinkINVENT_Exp3", smiles_generator, test_out_dir)


def test_benchmark_MolExp(smiles_generator, test_out_dir):
    """Test the MolExp benchmark."""
    _test_preset_benchmark("MolExp", smiles_generator, test_out_dir)


def test_benchmark_MolExpL(smiles_generator, test_out_dir):
    """Test the MolExpL benchmark."""
    _test_preset_benchmark("MolExpL", smiles_generator, test_out_dir)


def test_benchmark_MolExp_baseline(smiles_generator, test_out_dir):
    """Test the MolExp_baseline benchmark."""
    _test_preset_benchmark("MolExp_baseline", smiles_generator, test_out_dir)


def test_benchmark_MolExpL_baseline(smiles_generator, test_out_dir):
    """Test the MolExpL_baseline benchmark."""
    _test_preset_benchmark("MolExpL_baseline", smiles_generator, test_out_dir)

@pytest.mark.skipif(True, reason="3D benchmark not finalised")
def test_benchmark_3D_Benchmark(smiles_generator, test_out_dir):
    """Test the 3D_Benchmark benchmark."""
    _test_preset_benchmark("3D_Benchmark", smiles_generator, test_out_dir)


def _test_preset_benchmark(
    benchmark_name,
    smiles_generator,
    test_out_dir,
    budget=20,
    oracle_budget=True
    ):
    """Helper function to test a preset benchmark with parameterization."""
    # Skip if benchmark doesn't exist
    if benchmark_name not in MolScoreBenchmark.presets:
        pytest.skip(f"Benchmark '{benchmark_name}' not found in presets")
    
    try:
        msb = MolScoreBenchmark(
            model_name="test",
            output_dir=test_out_dir,
            budget=budget,  # Parameterized budget
            oracle_budget=oracle_budget,
            benchmark=benchmark_name
        )
    except Exception as e:
        pytest.fail(f"Failed to create benchmark '{benchmark_name}': {str(e)}")
    
    assert len(msb.configs) > 0, f"No configs loaded for benchmark {benchmark_name}"
    
    # Test all configs in the benchmark
    for task_context in msb:
        try:
            with task_context as scoring_function:
                while not scoring_function.finished:
                    smiles = smiles_generator.sample(BATCH_SIZE)
                    scores = scoring_function.score(smiles=smiles)
                    assert scores is not None, f"Scoring failed for {benchmark_name}"
                if not oracle_budget:
                    assert len(scoring_function.main_df) >= budget, f"Budget mismatch for {benchmark_name}"
                else:
                    assert len(scoring_function.exists_map) >= budget, f"Oracle budget mismatch for {benchmark_name}"
        except Exception as e:
            msg = str(e)
            if msg.startswith("__call__() missing 1 required positional argument:"):
                pytest.xfail(reason=msg)
            else:
                pytest.fail(f"Failed to run scoring for a config in '{benchmark_name}': {str(e)}") 
                   
    # Test metrics calculation if requested
    try:
        metrics = msb.summarize(overwrite=True)
        assert isinstance(metrics, list), f"Metrics for {benchmark_name} should be a list"
        assert len(metrics) > 0, f"Metrics list for {benchmark_name} should not be empty"
        assert len(metrics) == len(msb.configs), f"Metrics length should match number of configs for {benchmark_name}"
        assert os.path.exists(os.path.join(msb.output_dir, "results.csv")), f"results.csv for {benchmark_name} should exist"
    except Exception as e:
        pytest.fail(f"Failed to calculate metrics for '{benchmark_name}': {str(e)}")

# ----- Test custom benchmark directory which won't have metrics -----
def test_custom_benchmark(
    configs,
    smiles_generator,
    test_out_dir,
    budget=20,
    ):
    """Test a custom benchmark directory."""
    if configs is None:
        pytest.skip("No custom benchmark directory provided")
    
    if not os.path.isdir(configs):
        pytest.skip(f"Custom benchmark directory '{configs}' does not exist")
    
    try:
        msb = MolScoreBenchmark(
            model_name="test",
            output_dir=test_out_dir,
            budget=budget,
            custom_benchmark=configs
        )
    except Exception as e:
        pytest.fail(f"Failed to create benchmark from directory '{configs}': {str(e)}")
    
    assert len(msb.configs) > 0, f"No configs loaded from custom benchmark directory '{configs}'"
    
    # Test all configs in the custom benchmark
    for task_context in msb:
        try:
            with task_context as scoring_function:
                smiles = smiles_generator.sample(BATCH_SIZE)
                scores = scoring_function.score(smiles=smiles)
                assert scores is not None, f"Scoring failed for a config in {configs}"
        except Exception as e:
            msg = str(e)
            if msg.startswith("__call__() missing 1 required positional argument:"):
                pytest.xfail(reason=msg)
            else:
                pytest.fail(f"Failed to run scoring for a config in '{configs}': {str(e)}")
    
    # Should output default metrics
    metrics = msb.summarize(overwrite=True)
    assert isinstance(metrics, list), f"Metrics for custom benchmark should be a list"
    assert len(metrics) > 0, f"Metrics list for custom benchmark should not be empty"
    assert os.path.exists(os.path.join(msb.output_dir, "results.csv")), f"results.csv for custom benchmark should exist"


if __name__ == "__main__":
    pytest.main([__file__])