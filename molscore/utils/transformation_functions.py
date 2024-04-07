import numpy as np


def raw(x: float, **kwargs):
    """
    Dummy function to return metric 'as is'.
    :param x: Input value
    :return:
    """
    y = x
    return y


def norm(x: float, objective: str, max: float, min: float, **kwargs):
    """
    Normalize between maximum and minimum
    :param x: Input value
    :param objective: Maximize or minimize score [maximize, minimize]
    :param max: Maximum value for normalizing to (optional)
    :param min: Minimum value for normalizing to (optional)
    :param kwargs:
    :return:
    """
    try:
        if objective == "maximize":
            y = (x - min) / (max - min)

        elif objective == "minimize":
            y = (x - max) / (min - max)

        else:
            raise
    except ZeroDivisionError:
        y = 0.0
    return y


def lin_thresh(
    x: float, objective: str, upper: float, lower: float, buffer: float, **kwargs
):
    """
    Transform values using a linear threshold
    :param x: Input valid
    :param objective: Maximize, minimize or range [maximize, minimize, range]
    :param upper: Upper bound for transforming values ('range' and 'maximize' only)
    :param lower: Lower bound for transforming values ('range' and 'minimize' only)
    :param buffer: Buffer between thresholds which will be on a linear scale
    :param kwargs:
    :return:
    """
    if objective == "maximize":
        if x >= upper:
            y = 1.0
        elif x <= upper - buffer:
            y = 0.0
        else:
            y = (x - (upper - buffer)) / (upper - (upper - buffer))

    elif objective == "minimize":
        if x <= lower:
            y = 1.0
        elif x >= lower + buffer:
            y = 0.0
        else:
            y = (x - (lower + buffer)) / (lower - (lower + buffer))

    elif objective == "range":
        if lower <= x <= upper:
            y = 1.0
        else:
            if x <= lower - buffer:
                y = 0.0
            elif lower - buffer < x < lower:
                y = (x - (lower - buffer)) / (lower - (lower - buffer))
            elif x >= upper + buffer:
                y = 0.0
            else:
                y = (x - (upper + buffer)) / (upper - (upper + buffer))

    else:
        raise
    return y


def step(x: float, objective: str, upper: float, lower: float, **kwargs):
    """
    Transform values using a step transformer (threshold)
    :param x: Input value
    :param objective: Maximize, minimize or range [maximize, minimize, range]
    :param upper: Upper bound for transforming values ('range' and 'maximize' only)
    :param lower: Lower bound for transforming values ('range' and 'minimize' only)
    :param kwargs:
    :return:
    """
    if objective == "maximize":
        if x >= upper:
            y = 1.0
        else:
            y = 0.0

    elif objective == "minimize":
        if x <= lower:
            y = 1.0
        else:
            y = 0.0

    elif objective == "range":
        if lower <= x <= upper:
            y = 1.0
        else:
            y = 0.0

    else:
        raise
    return y


def gauss(x: float, objective: str, mu: float, sigma: float, **kwargs):
    """
    Transform values using a Gaussian transformer
    :param x: Input value
    :param objective: Maximize, minimize or range [maximize, minimize, range]
    :param mu: Mean
    :param sigma: Standard deviation
    :param kwargs:
    :return:
    """

    if objective == "maximize":
        if x >= mu:
            y = 1.0
        else:
            y = np.exp(-0.5 * np.power((x - mu) / sigma, 2.0))
    elif objective == "minimize":
        if x <= mu:
            y = 1.0
        else:
            y = np.exp(-0.5 * np.power((x - mu) / sigma, 2.0))
    elif objective == "range":
        y = np.exp(-0.5 * np.power((x - mu) / sigma, 2.0))
    else:
        raise
    return y


def sigmoid(
    x: float, objective: str, upper: float, lower: float, scale: float, **kwargs
):
    """
    Transform values using a sigmoid function
    :param x: Input value
    :param objective: Maximize, minimize or range [maximize, minimize, range]
    :param objective: Maximize, minimize or range [maximize, minimize, range]
    :param upper: Upper bound for transforming values ('range' and 'maximize' only)
    :param lower: Lower bound for transforming values ('range' and 'minimize' only)
    :param scale: Gradient of sigmoid function
    :param kwargs:
    :return:
    """

    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    if objective == "maximize":
        shift = upper - (5 / scale)
        y = _sigmoid(scale * (x - shift))
    elif objective == "minimize":
        shift = -lower - (5 / scale)
        y = _sigmoid(scale * (-x - shift))
    elif objective == "range":
        if lower <= x <= upper:
            y = 1.0
        else:
            if x > upper:
                shift = -upper - (5 / scale)
                y = _sigmoid(scale * (-x - shift))
            else:  # x < lower
                shift = lower - (5 / scale)
                y = _sigmoid(scale * (x - shift))
    else:
        raise
    return y


def plot_mod(mod, func_kwargs: dict):
    """
    Plot transformation functions
    :param mod: Modifier object
    :param func_kwargs: Keyword arguments for the modifier object
    :return:
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(3, 3))
    scale_max = max(v for k, v in func_kwargs.items() if k != "objective")
    scale_max = 10 ** np.ceil(np.log10(scale_max))  # Round up to nearest log10
    X = np.linspace(0, scale_max, 101)
    Y = [mod(x, **func_kwargs) for x in X]
    plt.plot(X, Y, label=func_kwargs)
    plt.xlabel("E.g input")
    plt.ylabel("E.g output")
    plt.title(mod.__name__)
    plt.grid()
    # plt.legend()
    return fig


def plot_mod_objectives(mod, non_objective_kwargs: dict):
    import matplotlib.pyplot as plt

    objectives = ["maximize", "minimize"]
    if mod.__name__ != "norm":
        objectives.append("range")

    X = np.linspace(0, 1, 101)

    fig, axes = plt.subplots(
        ncols=len(objectives),
        nrows=1,
        sharex=True,
        sharey=False,
        figsize=(6 * len(objectives), 4),
    )
    for objective, ax in zip(objectives, axes.flatten()):
        Y = [mod(x, objective=objective, **non_objective_kwargs) for x in X]
        ax.plot(X, Y, label=non_objective_kwargs)
        ax.set_xlabel("x")
        ax.set_ylabel("modified x")
        ax.set_title(objective)
        ax.legend()
    plt.suptitle(mod.__name__)
    plt.show()
    return
