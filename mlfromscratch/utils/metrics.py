import numpy as np


def root_mean_square_error(y, y_hat) -> float:
    """Root mean square error function. L2 Norm. Euclidean norm.

    Parameters:
        y (list): Ground truth (correct) target values.
        y_hat (list): Estimated target values.

    Returns:
        Result (float): A non-negative floating point value (the best value
        is 0.0), or an array of floating point values, one for each
        individual target.
    """

    y = np.asarray(y)
    y_hat = np.asarray(y_hat)

    return np.sqrt(sum(((y - y_hat)**2)) / len(y))


def mean_absolute_error(y, y_hat) -> float:
    """Mean absolute error function. L1 Norm. Manhattan norm.

    Parameters:
        y (list): Ground truth (correct) target values.
        y_hat (list): Estimated target values.

    Returns:
        Result (float): A non-negative floating point value (the best value
        is 0.0), or an array of floating point values, one for each
        individual target.
    """

    y = np.asarray(y)
    y_hat = np.asarray(y_hat)

    return sum(np.abs(y - y_hat))/len(y)


def euclidean_dist(x1, x2):

    if type(x1) is list:
        x1 = np.asarray(x1)
    if type(x2) is list:
        x2 = np.asarray(x2)

    dist = np.sqrt(sum((x1 - x2)**2))
    return dist
