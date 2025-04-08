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
    
from molscore.utils import mock_generator

MockGenerator = mock_generator.MockGenerator