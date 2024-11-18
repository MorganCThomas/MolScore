import os
import warnings

from moleval import resources
from molscore.scoring_functions.utils import check_exe

# get directory of this file
DIR = os.path.dirname(os.path.abspath(__file__))
# get directory of project and data
DATA_DIR = os.path.join(DIR, "../../data/")
PROJECT_DIR = os.path.join(DIR, "../../")

# Example paths

EXAMPLE_PDB_PATH = str(resources.files("moleval.tests.data") / "1a2g.pdb")
EXAMPLE_LIGAND_PATH = str(resources.files("moleval.tests.data") / "1a2g_ligand.sdf")

# Constants collected from the codebase
FORCEFIELD = "uff"
ADD_COORDS = True
ROUND_DIGITS = 2

# PATHS
# REDUCE_PATH = os.path.join(PROJECT_DIR, "bin/reduce")
if not check_exe("reduce"):
    warnings.warn(
        'reduce not found and posecheck may not work properly: Install with "conda install -c bioconda reduce"'
    )
    REDUCE_PATH = None
else:
    REDUCE_PATH = "reduce"
SPLIT_PATH = None  # os.path.join(PROJECT_DIR, "data/crossdocked_split_by_name.pkl")

# Docking params
DOCKING_METHOD = "smina"
SMINA_PATH = str(resources.files("moleval.metrics.posecheck.bin") / "smina.static")
EXHAUSTIVENESS = 8
BOX_SIZE = 25
