import matplotlib.pyplot as plt
import numpy as np

from mlfromscratch.src.unsupervised.KMeans import KMeans
from mlfromscratch.utils.metrics import silhouette_score


def kmeans():
    np.random.seed(42)  # for reproducibility

    p1 = np.random.rand(50, 2) * 10 + 1
    p2 = np.random.rand(50, 2) * 10 + 12

    p3 = np.random.rand(50, 2) * 5
    p3[:, 0] = p3[:, 0] + 15

    p4 = np.random.rand(50, 2) * 5
    p4[:, 1] = p4[:, 1] + 15

    points = np.concatenate([p1, p2, p3, p4])

    model = KMeans(4, seed=42)
    centroids, clusters = model.fit(points)

    map_colors = {0: '#1f77b4', 1: '#ff7f0e', 2:  '#007f0e', 3: '#df0a17'}

    score = round(silhouette_score(points, clusters, centroids), 5)
    plot(map_colors, clusters, points, centroids, score)


def plot(map_colors, clusters, points, centroids, score):
    colors = list(map_colors.values())

    point_colors = [map_colors[i] for i in clusters]

    # Range of values in the x axis
    plt.figure(figsize=(7, 5))
    ax = plt.subplot(111)
    # Scatter the points
    ax.scatter(points[:, 0], points[:, 1], c=point_colors, s=50, lw=0)
    ax.scatter(centroids[:, 0], centroids[:, 1], c=colors, s=100, lw=1)
    # Draw the decision boundary

    ax.set_xlabel('$x_1$', size=15)
    ax.set_ylabel('$x_2$', size=15)
    ax.tick_params(axis='both', which='both', bottom=False, left=False,
                   top=False, right=False, labelbottom=False, labelleft=False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('KMeans: Random Centroid Initialization')
    ax.legend(loc='lower right', scatterpoints=3)

    ax.text(0.5, 0.5, f'Silhouette score: {score}', fontsize=10, ha="center")

    plt.show()


if __name__ == "__main__":
    kmeans()
