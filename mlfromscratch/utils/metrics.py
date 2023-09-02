import numpy as np


def root_mean_square_error(y: list, y_hat: list) -> float:
    """Root mean square error function. L2 Norm. Euclidean norm.

    Parameters:
        y (list): Ground truth (correct) target values.
        y_hat (list): Estimated target values.

    Returns:
        Result (float): A non-negative floating point value (the best value
        is 0.0), or an array of floating point values, one for each
        individual target.
    """
    if type(y) is list:
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
    if type(y) is list:
        y = np.asarray(y)
        y_hat = np.asarray(y_hat)

    return sum(np.abs(y - y_hat))/len(y)


def euclidean_dist(x1, x2) -> float:
    """Compute the euclidean distance between two vectors x1, x2.

    Parameters:
        x1 (array): vector x1
        x2 (array): vector x2

    Returns:
        dist (float): The distances between the row vectors x1, x2
    """

    if type(x1) is list:
        x1 = np.asarray(x1)
    if type(x2) is list:
        x2 = np.asarray(x2)

    dist = np.sqrt(sum((x1 - x2)**2))

    return dist


def silhouette_score(points, clusters, centroids) -> float:
    """Compute the mean Silhouette Coefficient of all samples.

    The Silhouette Coefficient is calculated using the mean intra-cluster
    distance (``a``) and the mean nearest-cluster distance (``b``) for each
    sample.  The Silhouette Coefficient for a sample is ``(b - a) / max(a,
    b)``.  To clarify, ``b`` is the distance between a sample and the nearest
    cluster that the sample is not a part of.


    This function returns the mean Silhouette Coefficient over all samples.

    Parameters:
        points (array): the data points
        clusters (array): values with the labels of the points
        centroids (array): the centroids of the clusters
    Returns:
        silhouette_score (float): Mean Silhouette Coefficient for all samples.
    """

    if type(points) is list:
        points = np.asarray(points)
        clusters = np.asarray(clusters)
        centroids = np.asarray(centroids)

    silhouette_score = 0

    no_clusters = clusters.max() + 1

    for i in range(no_clusters):

        # Calculate a(i)
        indices = np.where(clusters == i)
        cluster_points = points[indices]
        dist = 0
        for c in cluster_points:
            dist += euclidean_dist(centroids[i], c)
        a_i = dist/len(cluster_points)

        # Calculate b(i)
        dist = []
        for c in centroids:
            dist.append(euclidean_dist(centroids[i], c))
        dist = np.asarray(dist)

        closest_centroid = np.argsort(dist)[1]
        indices = np.where(clusters == closest_centroid)
        cluster_points = points[indices]
        dist = 0
        for c in cluster_points:
            dist += euclidean_dist(centroids[i], c)
        b_i = dist/len(cluster_points)

        # Silhouette score of a single cluster
        s_i = (b_i - a_i) / max(b_i, a_i)

        silhouette_score += s_i

    silhouette_score = silhouette_score / no_clusters
    return silhouette_score
