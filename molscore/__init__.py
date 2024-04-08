__version__ = "1.5"
__all__ = ["MolScore", "MolScoreBenchmark", "MockGenerator"]

# Import lib resources depending on python version
import sys

if sys.version_info[1] < 9:
    import importlib_resources as resources
else:
    from importlib import resources

from molscore.manager import MolScore, MolScoreBenchmark
from molscore.utils import MockGenerator
