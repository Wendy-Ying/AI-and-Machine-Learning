import numpy as np

def initialize_centroids_kmeans_pp(data, k):
    """
    Initialize centroids using the KMeans++ algorithm.
    
    Parameters:
        data (numpy.ndarray): Input data, shape (n_samples, n_features).
        k (int): Number of clusters.
    
    Returns:
        numpy.ndarray: Initialized centroids, shape (k, n_features).
    """
    n_samples, n_features = data.shape
    centroids = []
    
    # Step 1: Randomly choose the first centroid
    first_index = np.random.randint(0, n_samples)
    centroids.append(data[first_index])
    
    for _ in range(1, k):
        # Step 2: Compute the squared distances from each sample to the nearest centroid
        distances = np.min([np.linalg.norm(data - c, axis=1)**2 for c in centroids], axis=0)
        
        # Step 3: Choose the next centroid with a probability proportional to the squared distance
        probabilities = distances / np.sum(distances)
        next_index = np.random.choice(n_samples, p=probabilities)
        centroids.append(data[next_index])
    
    return np.array(centroids)

def kmeans_pp(data, k, max_iters=100, tol=1e-4):
    """
    KMeans algorithm with KMeans++ initialization.
    
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
    # Initialize centroids using KMeans++ method
    centroids = initialize_centroids_kmeans_pp(data, k)
    
    for iteration in range(max_iters):
        # Compute distances from each sample to each centroid
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        
        # Assign each sample to the nearest centroid
        labels = np.argmin(distances, axis=1)
        
        # Update centroids as the mean of all samples in each cluster
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        
        # Check for convergence (if centroids move less than tol)
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

    # Apply KMeans++ clustering
    k = 3
    centroids, labels = kmeans_pp(data, k)

    # Visualize the results
    import matplotlib.pyplot as plt
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
    plt.title("KMeans++ Clustering")
    plt.legend()
    plt.show()
