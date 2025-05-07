__version__ = "1.9.2"
__all__ = ["MolScore", "MolScoreBenchmark", "MolScoreCurriculum", "MockGenerator"]

# Import lib resources depending on python version
import sys

if sys.version_info[1] < 9:
    import importlib_resources as resources
else:
    from importlib import resources

from molscore.manager import MolScore, MolScoreBenchmark, MolScoreCurriculum
from molscore.utils import MockGenerator
