from molscore.scoring_functions.glide import GlideDock, GlideDockFromROCS
from molscore.scoring_functions.smina import SminaDock
from molscore.scoring_functions.rocs import ROCS
from molscore.scoring_functions.oedock import FRED
from molscore.scoring_functions.tanimoto import TanimotoSimilarity
#from molscore.scoring_functions.reinvent_svm import ActivityModel
from molscore.scoring_functions.descriptors import MolecularDescriptors, RDKitDescriptors
from molscore.scoring_functions.isomer import Isomer
from molscore.scoring_functions.substructure_filters import SubstructureFilters
from molscore.scoring_functions.substructure_match import SubstructureMatch
from molscore.scoring_functions.sklearn_model import SKLearnModel, EnsembleSKLearnModel
from molscore.scoring_functions.rascore_xgb import RAScore_XGB
#from molscore.scoring_functions.chemprop import ChemPropModel

all_scoring_functions = [
    TanimotoSimilarity,
    Isomer,
    MolecularDescriptors,
    RDKitDescriptors, # Back compatability
    SubstructureFilters,
    SubstructureMatch,
    SKLearnModel,
    EnsembleSKLearnModel,
    RAScore_XGB,
    ROCS,
    GlideDock,
    GlideDockFromROCS,
    SminaDock,
    FRED
#    ChemPropModel
]
