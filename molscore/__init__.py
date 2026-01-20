from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("molscore")
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0"

__all__ = ["MolScore", "MolScoreBenchmark", "MolScoreCurriculum", "MockGenerator"]

# Import lib resources depending on python version
import sys

if sys.version_info[1] < 9:
    import importlib_resources as resources
else:
    from importlib import resources

from molscore.manager import MolScore, MolScoreBenchmark, MolScoreCurriculum
from molscore.utils import MockGenerator
