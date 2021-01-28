import numpy as np
import matplotlib.pyplot as plt


def raw(x: float, **kwargs):
    """
    Dummy function to return raw values or 'as is'.
    :param x:
    :return:
    """
    y = x
    return y


def norm(x: float, objective: str, max: float, min: float, **kwargs):
    """

    :param x:
    :param objective:
    :param max:
    :param min:
    :param kwargs:
    :return:
    """
    if objective == 'maximize':
        y = (x - min) / (max - min)

    elif objective == 'minimize':
        y = (x - max) / (min - max)

    else:
        print("Normalization objective must be either \'minimize\' or \'maximize\'")
        raise

    return y


def lin_thresh(x: float, objective: str, lower: float, upper: float, buffer: float, **kwargs):
    """

    :param x:
    :param objective:
    :param lower:
    :param upper:
    :param buffer:
    :param kwargs:
    :return:
    """
    if objective == 'maximize':
        if x >= upper:
            y = 1.0
        elif x <= upper-buffer:
            y = 0.0
        else:
            y = (x - (upper-buffer)) / (upper - (upper-buffer))

    elif objective == 'minimize':
        if x <= lower:
            y = 1.0
        elif x >= lower+buffer:
            y = 0.0
        else:
            y = (x - (lower+buffer)) / (lower - (lower+buffer))

    elif objective == 'range':
        if lower <= x <= upper:
               y = 1.0
        else:
            if x <= lower-buffer:
                y = 0.0
            elif lower-buffer < x < lower:
                y = (x - (lower-buffer)) / (lower - (lower-buffer))
            elif x >= upper+buffer:
                y = 0.0
            else:
                y = (x - (upper+buffer)) / (upper - (upper+buffer))

    else:
        print("linThresh objective must be either \'minimize\' or \'maximize\' or \'range\'")
        raise

    return y


def step(x: float, objective: str, lower: float, upper: float, **kwargs):
    if objective == 'maximize':
        if x >= upper:
            y = 1.0
        else:
            y = 0.0

    elif objective == 'minimize':
        if x <= lower:
            y = 1.0
        else:
            y = 0.0

    elif objective == 'range':
        if lower <= x <= upper:
            y = 1.0
        else:
            y = 0.0

    else:
        print("linThresh objective must be either \'minimize\' or \'maximize\' or \'range\'")
        raise

    return y


def gauss(x: float, objective: str, mu: float, sigma: float, **kwargs):
    """

    :param x:
    :param objective:
    :param mu:
    :param sigma:
    :param std:
    :param kwargs:
    :return:
    """

    if objective == 'maximize':
        if x >= mu:
            y = 1.0
        else:
            y = np.exp(-0.5 * np.power((x - mu) / sigma, 2.))
    elif objective == 'minimize':
        if x <= mu:
            y = 1.0
        else:
            y = np.exp(-0.5 * np.power((x - mu) / sigma, 2.))
    elif objective == 'range':
        y = np.exp(-0.5 * np.power((x - mu) / sigma, 2.))
    else:
        print("linThresh objective must be either \'minimize\' or \'maximize\' or \'range\'")
        raise

    return y


def plot_mod(mod, func_kwargs: dict):
    X = np.linspace(0, 1, 101)
    Y = [mod(x, **func_kwargs) for x in X]
    plt.plot(X, Y, label=func_kwargs)
    plt.xlabel('x')
    plt.ylabel('modified x')
    plt.title(mod.__name__)
    plt.legend()
    plt.show()
    return


def plot_mod_objectives(mod, non_objective_kwargs: dict):
    objectives = ['maximize', 'minimize']
    if mod.__name__ != 'norm':
        objectives.append('range')

    X = np.linspace(0, 1, 101)

    fig, axes = plt.subplots(ncols=len(objectives), nrows=1, sharex=True, sharey=False, figsize=(6*len(objectives), 4))
    for objective, ax in zip(objectives, axes.flatten()):
        Y = [mod(x, objective=objective, **non_objective_kwargs) for x in X]
        ax.plot(X, Y, label=non_objective_kwargs)
        ax.set_xlabel('x')
        ax.set_ylabel('modified x')
        ax.set_title(objective)
        ax.legend()
    plt.suptitle(mod.__name__)
    plt.show()
    return
