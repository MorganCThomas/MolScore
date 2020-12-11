from molscore.scoring_functions.glide import GlideDock, GlideDockFromROCS
from molscore.scoring_functions.rocs import ROCS
from molscore.scoring_functions.oedock import FRED
from molscore.scoring_functions.tanimoto import TanimotoSimilarity
#from molscore.scoring_functions.reinvent_svm import ActivityModel
from molscore.scoring_functions.descriptors import RDKitDescriptors
from molscore.scoring_functions.substructure_filters import SubstructureFilters
from molscore.scoring_functions.substructure_match import SubstructureMatch

all_scoring_functions = [
    ROCS,
    GlideDock,
    GlideDockFromROCS,
    FRED,
    RDKitDescriptors,
    TanimotoSimilarity,
    SubstructureFilters,
    SubstructureMatch
]
