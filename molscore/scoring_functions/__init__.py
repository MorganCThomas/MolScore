from molscore.scoring_functions.glide import GlideDock, GlideDockFromROCS
from molscore.scoring_functions.plants import PLANTSDock
from molscore.scoring_functions.gold import GOLDDock, ChemPLPGOLDDock, ASPGOLDDock, ChemScoreGOLDDock, GoldScoreGOLDDock
from molscore.scoring_functions.smina import SminaDock
from molscore.scoring_functions.rocs import ROCS
from molscore.scoring_functions.oedock import OEDock
from molscore.scoring_functions.similarity import MolecularSimilarity, TanimotoSimilarity
from molscore.scoring_functions.applicability_domain import ApplicabilityDomain
from molscore.scoring_functions.align3d import Align3D
#from molscore.scoring_functions.reinvent_svm import ActivityModel
from molscore.scoring_functions.pidgin import PIDGIN
from molscore.scoring_functions.descriptors import MolecularDescriptors, RDKitDescriptors
from molscore.scoring_functions.isomer import Isomer
from molscore.scoring_functions.substructure_filters import SubstructureFilters
from molscore.scoring_functions.substructure_match import SubstructureMatch
from molscore.scoring_functions.sklearn_model import SKLearnModel, EnsembleSKLearnModel
from molscore.scoring_functions.rascore_xgb import RAScore_XGB
#from molscore.scoring_functions.chemprop import ChemPropModel

all_scoring_functions = [
    MolecularSimilarity, 
    TanimotoSimilarity, # Back compatability
    MolecularDescriptors,
    RDKitDescriptors, # Back compatability
    Isomer,
    SubstructureFilters,
    SubstructureMatch,
    SKLearnModel,
    EnsembleSKLearnModel,
    PIDGIN,
    RAScore_XGB,
    ROCS,
    Align3D,
    GlideDock,
    GlideDockFromROCS,
    SminaDock,
    PLANTSDock,
    GOLDDock,
    ChemPLPGOLDDock,
    ASPGOLDDock,
    ChemScoreGOLDDock,
    GoldScoreGOLDDock,
    OEDock
#    ChemPropModel
]
