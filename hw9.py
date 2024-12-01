import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components  # Number of principal components to keep
        self.eigenvalues = None
        self.eigenvectors = None
        self.projection_matrix = None

    def fit(self, X):
        # Compute the covariance matrix
        cov_matrix = np.cov(X.T)
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        # Sort eigenvalues and eigenvectors in descending order
        eigen_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]
        eigen_pairs.sort(key=lambda k: k[0], reverse=True)
        # Select the top n_components eigenvectors
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.projection_matrix = np.hstack([eigen_pairs[i][1][:, np.newaxis] for i in range(self.n_components)])

    def transform(self, X):
        # Project the data onto the principal components
        return X.dot(self.projection_matrix)

    def reconstruct(self, X_pca):
        # Reconstruct the data from PCA space
        X_pca = self.transform(X_pca)
        return X_pca.dot(self.projection_matrix.T)

    def visualize(self, X_pca, y):
        # Visualize the projected data
        colors = ['r', 'b', 'g']
        markers = ['s', 'x', 'o']
        for label, color, marker in zip(np.unique(y), colors, markers):
            plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], 
                        c=color, label=label, marker=marker)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.title('PCA - Projected Data')
        plt.legend()
        plt.savefig('pca.png')
        plt.show()

    def plot_explained_variance(self):
        # Plot the cumulative explained variance for each component
        total_variance = np.sum(self.eigenvalues)
        explained_variance_ratio = self.eigenvalues / total_variance
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

        plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('PCA - Explained Variance')
        plt.savefig('pca_explained_variance.png')
        plt.show()


class LinearAutoencoder:
    def __init__(self, input_dim, encoding_dim, learning_rate=0.01, epochs=1000):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.W1 = np.random.randn(input_dim, encoding_dim) * 0.01
        self.W2 = np.random.randn(encoding_dim, input_dim) * 0.01
        self.b1 = np.zeros((1, encoding_dim))
        self.b2 = np.zeros((1, input_dim))
        self.losses = []  # Store the loss values for plotting the loss curve

    def encode(self, X):
        return np.dot(X, self.W1) + self.b1

    def decode(self, Z):
        return np.dot(Z, self.W2) + self.b2

    def fit(self, X):
        # Train the autoencoder
        for epoch in range(self.epochs):
            encoded = self.encode(X)
            decoded = self.decode(encoded)
            loss = np.mean((X - decoded) ** 2)  # Calculate mean squared error (MSE)
            self.losses.append(loss)  # Store loss value for later plotting
            
            # Backpropagation (gradient descent)
            dL_ddecoded = 2 * (decoded - X) / X.shape[0]
            dL_dW2 = np.dot(encoded.T, dL_ddecoded)
            dL_db2 = np.sum(dL_ddecoded, axis=0, keepdims=True)
            dL_dencoded = np.dot(dL_ddecoded, self.W2.T)
            dL_dW1 = np.dot(X.T, dL_dencoded)
            dL_db1 = np.sum(dL_dencoded, axis=0, keepdims=True)

            # Update weights and biases using gradient descent
            self.W1 -= self.learning_rate * dL_dW1
            self.b1 -= self.learning_rate * dL_db1
            self.W2 -= self.learning_rate * dL_dW2
            self.b2 -= self.learning_rate * dL_db2

    def transform(self, X):
        return self.encode(X)

    def reconstruct(self, X):
        encoded = self.encode(X)
        return self.decode(encoded)

    def visualize(self, X_encoded, y):
        # Visualize the encoded data (in 2D)
        colors = ['r', 'b', 'g', 'y', 'c']
        markers = ['s', 'x', 'o', '^', 'D']
        for label, color, marker in zip(np.unique(y), colors, markers):
            plt.scatter(X_encoded[y == label, 0], X_encoded[y == label, 1], 
                        c=color, label=f'Class {label}', marker=marker)
        
        plt.xlabel('Encoded Feature 1')
        plt.ylabel('Encoded Feature 2')
        plt.title('Autoencoder - Encoded Data')
        plt.legend()
        plt.savefig('autoencoder.png')
        plt.show()

    def plot_loss_curve(self):
        # Plot the loss curve of the autoencoder
        plt.plot(range(self.epochs), self.losses, label='Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Autoencoder - Loss Curve')
        plt.legend()
        plt.savefig('autoencoder_loss.png')
        plt.show()

def load_data():
    # Load and preprocess the wine dataset
    df_wine = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X = normalize_data(X)
    return X, y

def normalize_data(X):
    # Normalize the data to the range [0, 1]
    return (X - X.mean(axis=0)) / (X.max(axis=0) - X.min(axis=0))

def evaluate_model(model, X, y):
    # Evaluate the model by calculating the reconstruction error
    X_pca = model.transform(X)
    X_reconstructed = model.reconstruct(X)
    reconstruction_error = np.mean((X - X_reconstructed) ** 2)
    print(f"Reconstruction Error: {reconstruction_error}")
    # Visualize the transformed data
    model.visualize(X_pca, y)
    return reconstruction_error

def main():
    # Load the dataset
    X, y = load_data()

    # Create PCA instance and fit to the data
    pca = PCA(n_components=2)
    pca.fit(X)
    # Evaluate the PCA model
    evaluate_model(pca, X, y)
    # Plot the explained variance of PCA
    pca.plot_explained_variance()

    # Create Linear Autoencoder instance and fit to the data
    autoencoder = LinearAutoencoder(input_dim=X.shape[1], encoding_dim=5, learning_rate=0.005, epochs=30000)
    autoencoder.fit(X)
    # Evaluate the Autoencoder model
    evaluate_model(autoencoder, X, y)
    # Plot the loss curve of the autoencoder
    autoencoder.plot_loss_curve()

if __name__ == '__main__':
    main()
