import numpy as np
from mlfromscratch.src.unsupervised.KMeans import KMeans
from mlfromscratch.utils.metrics import silhouette_score

import warnings
import mlflow
import logging
import sys
import os

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def kmeans(no_clusters: int):

    np.random.seed(42)  # for reproducibility

    p1 = np.random.rand(50, 2) * 10 + 1
    p2 = np.random.rand(50, 2) * 10 + 12

    p3 = np.random.rand(50, 2) * 5
    p3[:, 0] = p3[:, 0] + 15

    p4 = np.random.rand(50, 2) * 5
    p4[:, 1] = p4[:, 1] + 15

    points = np.concatenate([p1, p2, p3, p4])

    model = KMeans(no_clusters, seed=42)
    centroids, clusters = model.fit(points)

    score = round(silhouette_score(points, clusters, centroids), 5)

    mlflow.log_param("k", k)
    mlflow.log_metric("silhouette_score", score)


if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    kmeans(k)
