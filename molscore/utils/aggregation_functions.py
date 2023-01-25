import numpy as np
import statistics as stats
from itertools import combinations
from scipy.stats import gmean as geometricmean

from molscore.utils.utils import ParetoFrontRank
from molscore.scoring_functions.utils import Fingerprints


def single(x, **kwargs):
    """
    Dummy function for single property optimization
    :param x: Vector of scores
    :return: Aggregate score bound between [0, 1]
    """
    return x


def wsum(x, w, **kwargs):
    """
    Weighted sum
    :param x: Vector of scores
    :param w: Vector of weights that should sum to 1
    :return: Aggregate score bound between [0, 1]
    """
    assert sum(w) == 1.0, "Weights must sum to 1"
    y = [xi*wi for xi, wi in zip(x, w)]
    return sum(y)


def gmean(x, **kwargs):
    """
    Geometric mean
    :param x: Vector of scores
    :return: Aggregate score bound between [0, 1]
    """
    y = geometricmean(x)
    return y


def amean(x, **kwargs):
    """
    Arithmetic mean
    :param x: Vector of scores
    :return: Aggregate score bound between [0, 1]
    """
    return stats.mean(x)


def pareto_pair(x, X, **kwargs):
    """
    Inspired from - 'De Novo Drug Design of Targeted Chemical Libraries Based on
    Artificial Intelligence and Pair-Based Multiobjective Optimization'
    https://pubs.acs.org/doi/10.1021/acs.jcim.0c00517
    Note: In this implementation scores r(xi) outside specified ranges are penalized according to the specified transformations functions.
    Note: The final fitness function is modified to within [0, 1], by equal contributions of reward R(x) and dominance P(x)
    :param x: Vector of scores
    :param X: Numpy matrix of reference scores (e.g., within the batch)
    :return: Aggregate score bound between [0, 1]
    """
    raise NotImplementedError
    N = len(x)
    # Compute R(x)
    R_x = sum(x)

    # Compute P(x)
    if R_x < N:
        P_x = 0.0

    elif R_x == N:
        # Compute m (number of molecules in X for which all r(xi) == 1)
        m = len(np.all(X == 1.0, axis=1))
        # Compute domination matrix for x
        domination_vector = np.zeros(len(x))
        for b in range(len(X)):
            # If 
            pass

        # Iterate over feature pairs
        domination = []
        for i, j in combinations(N, 2):
            # Compute dij (number of molecules in X that dominate over this molecule with respect to the i,j pairs)
            dij = sum(X[:, [i, j]].sum(axis=1) > (x[i] + x[j]))
            domination.append((m-dij)/m)
        # Compute average
        P_x = stats.mean(domination)    
    else:
        raise ValueError(f"Aggregate score should not sum to more than the number of features")
    
    print(f'R(x)={R_x}')
    print(f'P(x)={P_x}')
    
    # Compute final fitness S(x) = R(x) + P(x)
    S_x = stats.mean(R_x) + (P_x*0.5)
 
    return S_x 


class ParetoFront:
    """
    Pareto front according to DrugEx v2
    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00561-9
    """
    X = None
    fps = None
    rank = None

    @classmethod
    def pareto_front(cls, x, X: np.ndarray, batch_smiles: list, thresh: float = 0.5, **kwargs):
        """
        Implementation of Pareto Front score from 'DrugEx v2 - de novo design of drug molecules by 
        Pareto-based multi-objective reinforcement learning in polypharmacology'
        :param x: Vector of scores
        :param X: Scores for all molecules within a batch
        :param thresh: Threshold to define desirable or undesirable molecules
        :param batch_smiles: Smiles for all molecules within a batch
        """
        # Update class to avoid computing every query x in the same batch
        if (cls.X is None) or (cls.X != X).any():
            cls.X = X
            cls.fps = [Fingerprints.get(smi, name='ECFP6', nBits=2048) for smi in batch_smiles]
            cls.rank = ParetoFrontRank(cls.X, cls.fps)
        # Compute reward for molecule
        x_desirable = np.all(x >= thresh) # All xi are above/equal to threshold
        n_desirable = np.all(X >= thresh, axis=1).sum() # Number of X desirable
        n_undesirable = len(X) - n_desirable # Number of X undesirable
        i = np.where(np.all(X == x, axis=1))[0][0]  # Index of x in X
        k = cls.rank.index(i) # Index of x in ranks
        if x_desirable:
            score = (1-thresh) + ((k-n_undesirable)/n_desirable)
        else:
            score = k / (2*n_undesirable)
        return score

