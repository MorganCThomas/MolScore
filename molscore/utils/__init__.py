from molscore.utils.transformation_functions import raw, norm, step, gauss, lin_thresh, sigmoid
from molscore.utils.aggregation_functions import single, wsum, prod, wprod, gmean, amean, ParetoFront, DynamicSum, DynamicProd

all_score_modifiers = [
    raw,
    norm,
    step,
    gauss,
    lin_thresh,
    sigmoid
]

all_score_methods = [
    single,
    amean,
    gmean,
    wsum,
    prod,
    wprod,
    DynamicSum.auto_wsum,
    DynamicProd.auto_wprod,
    ParetoFront.pareto_front,
]
