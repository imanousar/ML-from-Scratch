import numpy as np
from typing import Tuple


class LinearRegression:

    def __init__(self, random_seed: int = 42):
        """
        Constructor that initializes the LinearRegression
        object with random weights.

        Args:
            random_seed (int): Random seed for weight initialization.


        """
        self.m, self.b = self.initialize(random_seed)

    def initialize(self, seed: int) -> Tuple[float, float]:
        """
        Method that initializes the weights to random values.

         Args:
            seed (int): Random seed for weight initialization.

        Returns:
            A tuple containing the slope and y-intercept of the linear
            regression model.

        """
        np.random.seed(seed)
        m = np.random.random()
        b = np.random.random()
        return m, b

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Method that makes predictions for a number of points.

        Args:
            x (np.ndarray): Input array of shape (n_samples, n_features).

        Returns:
            A numpy array of shape (n_samples, ) containing the predicted
            y-values.

        """
        return self.m * x + self.b

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Method that handles the whole training procedure.

        Args:
            x (np.ndarray): Input array of shape (n_samples, n_features).
            y (np.ndarray): Output array of shape (n_samples, ).

        Returns:
            None
        """

        if x.shape != y.shape:
            raise ValueError("x and y should have the same shape")

        N = x.shape[0]

        nom = N*np.dot(x, y) - x.sum()*y.sum()
        den = N * np.dot(x, x) - np.dot(x.sum(), x.sum())
        self.m = nom/den
        self.b = (y.sum() - self.m * x.sum())/N

        return None
