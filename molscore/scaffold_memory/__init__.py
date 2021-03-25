from molscore.scaffold_memory.ScaffoldFilter import IdenticalMurckoScaffold, IdenticalTopologicalScaffold,  \
    CompoundSimilarity, ScaffoldSimilarityAtomPair, ScaffoldSimilarityECFP

all_scaffold_filters = [
    IdenticalMurckoScaffold,
    IdenticalTopologicalScaffold,
    CompoundSimilarity,
    ScaffoldSimilarityAtomPair,
    ScaffoldSimilarityECFP
]