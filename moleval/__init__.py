from molscore._version import __version__

__all__ = [
    "MockGenerator",
    "resources",
]

# Import lib resources depending on python version
import sys

if sys.version_info[1] < 9:
    import importlib_resources as resources
else:
    from importlib import resources
    
from moleval.mock_generator import MockGenerator