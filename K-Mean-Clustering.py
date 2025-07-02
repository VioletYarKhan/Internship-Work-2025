import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import random


def generate_particles(n_particles=10000):
    np.random.seed(random.randint(1, 1000))
    return np.random.rand(n_particles, 3)


def normalize_particles(particles):
    scaler = StandardScaler()
    return scaler.fit_transform(particles)


def balanced_kmeans(X, k, max_iter=500):
    # Partition particles into k equal-sized clusters using a balanced KMeans approach.

    # Parameters:
    #     X: np.ndarray of shape (N, 3) - the normalized 3D points.
    #     k: int - number of clusters.
    #     max_iter: int - number of iterations.

    # Returns:
    #     labels: np.ndarray of shape (N,) - cluster assignment for each point.
    #     centroids: np.ndarray of shape (k, 3) - centroid of each cluster.
    
    N = len(X)
    group_size = N // k
    assert N % k == 0, ValueError("Number of particles must be divisible by k for equal-sized clusters.")

    centroids = X[np.random.choice(N, k, replace=False)]
    labels = np.zeros(N, dtype=int)

    for iteration in range(max_iter):
        # Compute cost matrix: distances from each point to each centroid
        cost_matrix = pairwise_distances(X, centroids)

        pairs = [(i, j, cost_matrix[i, j]) for i in range(N) for j in range(k)]
        pairs.sort(key=lambda x: x[2])
        new_labels = -np.ones(N, dtype=int)
        cluster_counts = np.zeros(k, dtype=int)
        assigned = np.zeros(N, dtype=bool)
        for i, j, l in pairs:
            if not assigned[i] and cluster_counts[j] < group_size:
                new_labels[i] = j
                cluster_counts[j] += 1
                assigned[i] = True
            if assigned.all():
                break
        labels = new_labels
        # Recompute centroids
        for i in range(k):
            centroids[i] = X[labels == i].mean(axis=0)
    return labels, centroids


def visualize_clusters(X, labels, title="3D Particle Clustering"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='tab20', s=20)
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax, label='Cluster ID')
    plt.show()


num_particles = 10000
num_clusters = 10

print(f"Generating {num_particles} 3D particles...")
particles = generate_particles(num_particles)

print("Normalizing particles...")
normalized_particles = normalize_particles(particles)

print(f"Clustering into {num_clusters} equal-sized groups...")
labels, centroids = balanced_kmeans(normalized_particles, k=num_clusters)

print("Visualizing result...")
visualize_clusters(normalized_particles, labels)

print("Done.")