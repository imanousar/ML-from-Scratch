import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mlfromscratch.src.unsupervised.KMeans import KMeans
from mlfromscratch.utils.metrics import silhouette_score
import random


def main():
    # Title
    st.title("KMeans Clustering")
    st.write("Here is an example of a KMeans model.")

    # Sidebar
    st.sidebar.markdown("## Controls")
    st.sidebar.markdown(
        "You can **change** the values to change the *results*.")

    k = st.sidebar.slider('k', min_value=1,
                          max_value=10,  value=4, step=1)
    iterations = st.sidebar.slider('Iterations', min_value=1,
                                   max_value=15,  value=1, step=1)

    # Model Config

    np.random.seed(42)  # for reproducibility

    p1 = np.random.rand(50, 2) * 10 + 1
    p2 = np.random.rand(50, 2) * 10 + 12

    p3 = np.random.rand(50, 2) * 5
    p3[:, 0] = p3[:, 0] + 15

    p4 = np.random.rand(50, 2) * 5
    p4[:, 1] = p4[:, 1] + 15

    points = np.concatenate([p1, p2, p3, p4])

    with st.echo():
        model = KMeans(k=k, seed=42, max_iter=iterations)
        centroids, clusters = model.fit(points)

    score = round(silhouette_score(points, clusters, centroids), 5)

    # Plot
    map_colors = make_colors(int(k))
    try:
        plot(map_colors, clusters, points, centroids, score)
    except:
        pass


def plot(map_colors, clusters, points, centroids, score):
    colors = list(map_colors.values())
    point_colors = [map_colors[i] for i in clusters]

    # Range of values in the x axis
    fig = plt.figure(figsize=(7, 5))
    ax = plt.subplot(111)
    # Scatter the points
    plt.scatter(points[:, 0], points[:, 1], c=point_colors, s=50, lw=0)
    plt.scatter(centroids[:, 0], centroids[:, 1], c=colors, s=100, lw=1)
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

    st.pyplot(fig)


@st.cache
def make_colors(no_clusters):
    map_colors = {}
    for i in range(no_clusters):
        np.random.seed(42)
        random_number = random.randint(0, 16777215)
        hex_number = str(hex(random_number))
        hex_number = '#' + hex_number[2:]
        map_colors[i] = hex_number
    return map_colors


if __name__ == '__main__':
    main()
