import numpy as np


class LinearRegression:

    def __init__(self, random_seed=42):

        self.m, self.b = self.initialize(random_seed)

    def initialize(self, seed):
        """
        Method that initializes the weights to random values.
        """
        np.random.seed(seed)
        m = np.random.random()
        b = np.random.random()
        return m, b

    def predict(self, x):
        """
        Method that makes predictions for a number of points.
        """
        return self.m * x + self.b

    def fit(self, x, y):
        """
        Method that handles the whole training procedure.
        """
        N = x.shape[0]

        nom = N*np.dot(x, y) - x.sum()*y.sum()
        den = N * np.dot(x, x) - np.dot(x.sum(), x.sum())
        self.m = nom/den
        self.b = (y.sum() - self.m * x.sum())/N

        return None
