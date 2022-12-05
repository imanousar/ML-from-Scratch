import numpy as np
from mlfromscratch.utils.metrics import euclidean_dist


class KMeans:
    """ K-Means clustering

    Parameters:
        k (int): Number of clusters
        max_iter (int): Maximum number of iterations of the k-means algorithm
        for a single run.
        seed (int): A seed for the initialization of weights.

    Returns:
        None (none): None
    """

    def __init__(self, k=1, max_iter=5, seed=None):

        self.k = k
        self.max_iter = max_iter
        self.seed = seed

    def initialize(self, points):
        """
        Method that initializes the centroids to random values.
        """
        np.random.seed(self.seed)
        self.centroids = np.random.rand(self.k, 2) * points.max()

    def calculate_distances(self, points):

        distances = np.zeros((len(self.centroids), len(points)), dtype='float')
        for i, centroid in enumerate(self.centroids):
            for j, point in enumerate(points):
                distances[i, j] = euclidean_dist(centroid, point)

        return distances

    def update_clusters(self, distances):

        clusters = np.argmin(distances, axis=0)

        return clusters

    def update_centroids(self, clusters, points):

        size = max(clusters)+1

        for i in range(size):
            indices = np.where(clusters == i)
            cluster_points = points[indices]
            self.centroids[i] = cluster_points.mean(axis=0)

        return self.centroids

    def fit(self, x):
        """
        Compute k-means clustering
        """
        self.initialize(x)

        for i in range(self.max_iter):

            distances = self.calculate_distances(x)
            clusters = self.update_clusters(distances)
            self.centroids = self.update_centroids(clusters, x)

        return self.centroids, clusters

    def predict(self, x):

        distances = self.calculate_distances(self.centroids, x)
        self.clusters = self.update_clusters(distances)

        return self.clusters
