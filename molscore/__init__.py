__version__ = "1.4"

# Import lib resources
import sys
if sys.version_info[1] < 9:
    from importlib import resources
else:
    import importlib_resources as resources 

# Import managers 
from molscore.manager import MolScore, MolScoreBenchmark