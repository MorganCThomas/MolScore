import os
from molscore.utils.score_modifiers import norm, step, gauss, lin_thresh
from molscore.utils.score_methods import single, wsum, gmean, amean, pareto_pair

all_score_modifiers = [
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
