import statistics as stats
from scipy.stats import gmean as geometricmean


def single(X, **kwargs):
    """
    Dummy function for single property optimization
    :param X:
    :return:
    """
    return X


def wsum(X, W, **kwargs):
    """
    Weighted sum
    :param X:
    :param W:
    :return:
    """

    Y = [x*w for x, w in zip(X, W)]
    y = sum(Y)
    return y


def gmean(X, **kwargs):
    """
    Geometric mean
    :param X:
    :param kwargs:
    :return:
    """
    y = geometricmean(X)
    return y


def amean(X, **kwargs):
    """
    Arithmetic mean
    :param X:
    :param kwargs:
    :return:
    """
    y = stats.mean(X)
    return y


def pareto_pair(X, df, **kwargs):
    """
    Inspired from - 'De Novo Drug Design of Targeted Chemical Libraries Based on
     Artificial Intelligence and Pair-Based Multiobjective Optimization
     https://pubs.acs.org/doi/10.1021/acs.jcim.0c00517'.

    :param X:
    :param df:
    :param kwargs:
    :return:
    """
    # TODO
    raise NotImplementedError
