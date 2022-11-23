import numpy as np


class UniLinearRegression:

    def __init__(self, epochs=100, learning_rate=0.0001, random_seed=42):
        """ Univariate Linear Regression class

        Parameters:
        ----------
        epoch : int
            The number of epochs.
        learning_rate : float
            The learning rate of the algorithm.
        random_seed : int
            A seed for the initialization of weights.

        Returns:
        ----------
        None
        """

        self.epochs = epochs
        self.lr = learning_rate
        self.a, self.b = self.initialize(random_seed)
        self.a_history = []
        self.b_history = []

    def initialize(self, seed):
        """
        Method that initializes the weights to random values.
        """
        np.random.seed(seed)
        a = np.random.random()
        b = np.random.random()
        return a, b

    def predict(self, x):
        """
        Method that makes predictions for a number of points.
        """
        return self.a * x + self.b

    def cost(self, x, y):
        """
        Method that calculates the cost of the prediction (MSE).
        """
        y_hat = self.predict(x)
        return sum(((y - y_hat)**2)) / len(y)

    def update(self, x, y):
        """
        Method that runs one iteration of gradient descent and updates the
        class' weights
        """
        y_hat = self.predict(x)
        da = - (2 / len(x)) * sum(x * (y - y_hat))
        db = - (2 / len(x)) * sum(y - y_hat)
        self.a -= (self.lr * da)
        self.b -= (self.lr * db)

        return None

    def fit(self, x, y):
        """
        Method that handles the whole training procedure.
        """
        for ep in range(self.epochs):
            self.a_history.append(self.a)
            self.b_history.append(self.b)
            self.update(x, y)

        return None
