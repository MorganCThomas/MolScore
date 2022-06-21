from molscore.utils.transformation_functions import raw, norm, step, gauss, lin_thresh
from molscore.utils.aggregation_functions import single, wsum, gmean, amean, pareto_pair

all_score_modifiers = [
    raw,
    norm,
    step,
    gauss,
    lin_thresh
]

all_score_methods = [
    single,
    amean,
    gmean,
    wsum,
    pareto_pair
]
