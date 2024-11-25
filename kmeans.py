import numpy as np

def kmeans(data, k, max_iters=100, tol=1e-4):
    """
    A pure numpy implementation of the KMeans algorithm.

    Parameters:
        data (numpy.ndarray): Input data, shape (n_samples, n_features).
        k (int): Number of clusters.
        max_iters (int): Maximum number of iterations.
        tol (float): Tolerance for convergence; stops when the centroid movement is below this value.

    Returns:
        tuple: (centroids, labels)
            - centroids (numpy.ndarray): Cluster centroids, shape (k, n_features).
            - labels (numpy.ndarray): Cluster labels for each sample, shape (n_samples,).
    """
    # Step 1: Initialize centroids by randomly selecting k samples from the data
    n_samples, n_features = data.shape
    random_indices = np.random.choice(n_samples, k, replace=False)
    centroids = data[random_indices]

    for iteration in range(max_iters):
        # Step 2: Compute distances from each sample to each centroid
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        
        # Step 3: Assign each sample to the nearest centroid
        labels = np.argmin(distances, axis=1)
        
        # Step 4: Recompute centroids as the mean of all samples in each cluster
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        
        # Step 5: Check for convergence (if centroids move less than tol)
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        
        centroids = new_centroids

    return centroids, labels

# Test the implementation
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    data = np.vstack([
        np.random.normal(loc=0.0, scale=1.0, size=(50, 2)),
        np.random.normal(loc=5.0, scale=1.0, size=(50, 2)),
        np.random.normal(loc=10.0, scale=1.0, size=(50, 2))
    ])

    # Apply KMeans clustering
    k = 3
    centroids, labels = kmeans(data, k)

    # Visualize the results
    import matplotlib.pyplot as plt
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
    plt.title("KMeans Clustering")
    plt.legend()
    plt.show()
