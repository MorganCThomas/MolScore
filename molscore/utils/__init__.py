__all__ = ["MockGenerator", "all_score_modifiers", "all_score_methods"]

from molscore.utils.aggregation_functions import (
    DynamicProd,
    DynamicSum,
    ParetoFront,
    amean,
    gmean,
    prod,
    single,
    wprod,
    wsum,
)
from molscore.utils.mock_generator import MockGenerator
from molscore.utils.transformation_functions import (
    gauss,
    lin_thresh,
    norm,
    raw,
    sigmoid,
    step,
)

all_score_modifiers = [raw, norm, step, gauss, lin_thresh, sigmoid]

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
