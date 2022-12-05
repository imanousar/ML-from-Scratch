import numpy as np


# TODO: Add stratify option
def train_test_split(data, test_ratio, seed=False):
    """description

    Parameters:
        data ():
        test_ratio ():
        seed ():

    Returns:

    """
    if seed:
        np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]


# TODO: Min max, normal scaling
def scaler(data):
    pass
