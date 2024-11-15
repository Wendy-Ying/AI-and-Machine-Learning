from scipy.spatial import KDTree
import numpy as np

class KNN:
    def __init__(self, k=3):
        """
        Initialize the KNN classifier.

        Parameters:
        k (int): The number of neighbors to consider.
        """
        self.k = k
        self.kdtree = None

    def fit(self, X, y):
        """
        Fit the KNN model using the training data.

        Parameters:
        X (array-like): Training data features of shape (n_samples, n_features).
        y (array-like): Training data labels of shape (n_samples).
        """
        self.X_train = X
        self.y_train = y
        self.kdtree = KDTree(X)

    def predict_multiple(self, X):
        """
        Predict the labels for multiple data points.

        Parameters:
        X (array-like): Data points to predict, shape (n_samples, n_features).

        Returns:
        array: Predicted labels for each input sample, shape (n_samples).
        """
        predictions = [self.predict_single(x) for x in X]
        return np.array(predictions)

    def predict_single(self, x):
        """
        Predict the label for a single data point.

        Parameters:
        x (array-like): A single data point feature, shape (n_features,).

        Returns:
        The predicted label for the input data point.
        """
        # Find k nearest neighbors
        dist, idx = self.kdtree.query(x, k=self.k, p=2)

        # If idx is a 1D array (i.e., querying a single point), convert it to a list
        if idx.ndim == 1:
            idx = [idx]

        # Get the labels of the neighbors
        neighbors_labels = [self.y_train[i] for i in idx[0]]

        # Determine the most common label among the neighbors
        prediction = max(set(neighbors_labels), key=neighbors_labels.count)
        return prediction

if __name__ == "__main__":
    # Sample training data
    X_train = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    # Create KNN instance and fit the model
    knn = KNN(k=3)
    knn.fit(X_train, y_train)

    # Sample test data
    X_test = np.array([[1, 1], [3, 1], [5, 5]])

    # Make predictions
    predictions = knn.predict_multiple(X_test)
    print("Predicted labels:", predictions)
