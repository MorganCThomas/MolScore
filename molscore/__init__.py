__version__ = "1.4"

# Import lib resources
import sys
if sys.version_info[1] < 9:
    import importlib_resources as resources 
else:
    from importlib import resources

# Import managers 
from molscore.manager import MolScore, MolScoreBenchmark