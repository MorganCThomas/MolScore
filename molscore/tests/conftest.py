from pathlib import Path
import pytest
from molscore import MockGenerator
    
@pytest.fixture
def test_out_dir():
    return Path(__file__).parent / "test_out"

@pytest.fixture
def ref_smiles():
    mg = MockGenerator(seed_no=123)
    return mg.sample(5)

@pytest.fixture
def in_smiles():
    mg = MockGenerator(seed_no=321, augment_invalids=True, augment_duplicates=False, augment_none=True)
    return mg.sample(10)

@pytest.fixture
def smiles_generator():
    return MockGenerator(seed_no=231, augment_invalids=True, augment_duplicates=True, augment_none=True)

def pytest_addoption(parser):
    parser.addoption(
        "--configs",
        action="store",
        nargs="+",
        help="One or more configuration definitions, paths, or directories.",
    )
    parser.addoption(
        "--full", action="store_true", default=False, help="Run full parameter grid"
    )
    
def pytest_configure(config):
    config.addinivalue_line("markers", "full: mark test as part of full parameter grid")