import numpy as np

from molscore.scoring_functions.utils import SimilarityMeasures


def dominated(x1: np.ndarray, x2: np.ndarray) -> bool:
    """Check is x1 is dominated by x2 according to DrugEx v2 defintion"""
    return np.all(x1 <= x2) & np.any(x1 < x2)


def non_dominated_sort(X: np.ndarray) -> list:
    """
    The non-dominated sorting algorithm according to DrugEx v2 a.k.a NSGAII
    :param X: m x n scorig matrix, where m is the number of samples and n is the number of objectives.
    :return: a list of Pareto fronts, in which the dominated solutions are on the top, and non-dominated solutions are on the bottom.
    """
    domina = [[] for _ in range(len(X))]  # List of dominating solutions over each entry
    count = np.zeros(
        len(X), dtype=int
    )  # Count of how many times this solution dominates another
    ranks = np.zeros(len(X), dtype=int)  # Zeros to start
    front = []  # Index of non-dominating solutions as front
    for p, ind1 in enumerate(X):
        for q in range(p + 1, len(X)):
            ind2 = X[q]
            if dominated(ind1, ind2):
                domina[p].append(q)
                count[q] += 1
            elif dominated(ind2, ind1):
                domina[q].append(p)
                count[p] += 1

        if count[p] == 0:
            ranks[p] = 0
            front.append(p)

    fronts = [np.sort(front)]
    i = 0
    while len(fronts[i]) > 0:
        temp = []
        for f in fronts[i]:
            for d in domina[f]:
                count[d] -= 1
                if count[d] == 0:
                    ranks[d] = i + 1
                    temp.append(d)
        i = i + 1
        fronts.append(np.sort(temp))
    del fronts[len(fronts) - 1]  # Delete empty last front
    return fronts


def ParetoFrontRank(X: np.ndarray, fps: list) -> list:
    """
    Tanimoto sorting of each pareto frontier according to DrugEx v2.
    :param X: m x n scorig matrix, where m is the number of samples and n is the number of objectives.
    :param fps: List of RDKit fingerprints for all the molecules
    :return: m-d vector as the index of well-ranked solutions.
    """
    fronts = non_dominated_sort(X)
    rank = []
    for i, front in enumerate(fronts):
        fp = [fps[f] for f in front]
        if (len(front) > 2) and None not in fp:
            dist = np.zeros(len(front))
            for j in range(len(front)):
                tanimoto = 1 - np.array(
                    SimilarityMeasures.get("Tanimoto", bulk=True)(fp[j], fp)
                )
                order = tanimoto.argsort()
                dist[order[0]] += 0
                dist[order[-1]] += 10**4
                for k in range(1, len(order) - 1):
                    dist[order[k]] += tanimoto[order[k + 1]] - tanimoto[order[k - 1]]
            fronts[i] = front[dist.argsort()]
        rank.extend(fronts[i].tolist())
    return rank