from molscore.utils.transformation_functions import raw, norm, step, gauss, lin_thresh
from molscore.utils.aggregation_functions import single, wsum, gmean, amean, ParetoFront, DynamicSum

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
    DynamicSum.dynamic_wsum,
    ParetoFront.pareto_front,
]
