from molscore.scaffold_memory.ScaffoldFilter import IdenticalMurckoScaffold, IdenticalTopologicalScaffold,  \
    CompoundSimilarity, ScaffoldSimilarityAtomPair, ScaffoldSimilarityECFP
from molscore.scaffold_memory.ScaffoldMemory import ScaffoldMemory

all_scaffold_filters = [
    IdenticalMurckoScaffold,
    IdenticalTopologicalScaffold,
    CompoundSimilarity,
    ScaffoldSimilarityAtomPair,
    ScaffoldSimilarityECFP
]