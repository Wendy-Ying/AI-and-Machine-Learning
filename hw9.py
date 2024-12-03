import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components  # Number of principal components to retain
        self.eigenvalues = None
        self.eigenvectors = None
        self.projection_matrix = None

    def fit(self, X):
        # Compute covariance matrix and eigenvalues/eigenvectors
        cov_matrix = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        # Sort eigenvalues and eigenvectors in descending order
        eigen_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]
        eigen_pairs.sort(key=lambda k: k[0], reverse=True)
        # Select the top n_components eigenvectors
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.projection_matrix = np.hstack([eigen_pairs[i][1][:, np.newaxis] for i in range(self.n_components)])

    def transform(self, X):
        # Project data onto the principal components
        return X.dot(self.projection_matrix)

    def reconstruct(self, X_pca):
        # Reconstruct original data from PCA space
        X_pca = self.transform(X_pca)
        return X_pca.dot(self.projection_matrix.T)

    def visualize(self, X_pca, y):
        plt.figure()
        # Visualize PCA-reduced data in 2D
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

    def plot_explained_variance(self):
        # Plot cumulative explained variance
        total_variance = np.sum(self.eigenvalues)
        explained_variance_ratio = self.eigenvalues / total_variance
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

        plt.figure()
        plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('PCA - Explained Variance')
        plt.savefig('pca_explained_variance.png')


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
        self.losses = []  # Store loss values

    def encode(self, X):
        return np.dot(X, self.W1) + self.b1

    def decode(self, Z):
        return np.dot(Z, self.W2) + self.b2

    def fit(self, X):
        # Train the autoencoder using gradient descent
        for epoch in range(self.epochs):
            encoded = self.encode(X)
            decoded = self.decode(encoded)
            loss = np.mean((X - decoded) ** 2)  # Calculate reconstruction loss
            self.losses.append(loss)  # Store loss value
            
            # Backpropagation (gradient descent)
            dL_ddecoded = 2 * (decoded - X) / X.shape[0]
            dL_dW2 = np.dot(encoded.T, dL_ddecoded)
            dL_db2 = np.sum(dL_ddecoded, axis=0, keepdims=True)
            dL_dencoded = np.dot(dL_ddecoded, self.W2.T)
            dL_dW1 = np.dot(X.T, dL_dencoded)
            dL_db1 = np.sum(dL_dencoded, axis=0, keepdims=True)

            # Update weights and biases
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
        plt.figure()
        # Visualize the encoded data in 2D
        colors = ['r', 'b', 'g', 'y', 'c']
        markers = ['s', 'x', 'o', '^', 'D']
        for label, color, marker in zip(np.unique(y), colors, markers):
            plt.scatter(X_encoded[y == label, 0], X_encoded[y == label, 1], 
                        c=color, label=f'Class {label}', marker=marker)
        plt.xlabel('Encoded Feature 1')
        plt.ylabel('Encoded Feature 2')
        plt.title('Linear Autoencoder - Encoded Data')
        plt.legend()
        plt.savefig('linear_autoencoder.png')

    def plot_loss_curve(self):
        # Plot the loss curve
        plt.figure()
        plt.plot(range(self.epochs), self.losses, label='Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Linear Autoencoder - Loss Curve')
        plt.legend()
        plt.savefig('linear_autoencoder_loss.png')

class NonlinearAutoencoder:
    def __init__(self, input_dim, hidden_dims, output_dim, learning_rate=0.01, epochs=1000):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims  # List of hidden layer dimensions
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.weights.append(np.random.randn(prev_dim, hidden_dim) * 0.01)
            self.biases.append(np.zeros((1, hidden_dim)))
            prev_dim = hidden_dim
        # Decoder weights (reverse the process)
        self.weights.append(np.random.randn(prev_dim, output_dim) * 0.01)
        self.biases.append(np.zeros((1, output_dim)))

        self.losses = []  # Store loss values

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def mse_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def forward(self, X):
        A = X
        activations = [A]
        pre_activations = []
        
        # Forward pass through hidden layers
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            Z = np.dot(A, W) + b
            A = self.relu(Z)
            pre_activations.append(Z)
            activations.append(A)
        
        # Decoder output
        Z = np.dot(A, self.weights[-1]) + self.biases[-1]
        activations.append(Z)
        
        return activations

    def backward(self, X, activations):
        m = X.shape[0]  # Number of training examples
        # Backpropagate the error
        dZ = activations[-1] - X  # Derivative of loss with respect to output
        dW = np.dot(activations[-2].T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        
        # Update decoder weights
        self.weights[-1] -= self.learning_rate * dW
        self.biases[-1] -= self.learning_rate * db
        
        # Backpropagate to hidden layers
        for i in range(len(self.hidden_dims)-1, -1, -1):
            dA = np.dot(dZ, self.weights[i+1].T)
            dZ = dA * self.relu_derivative(activations[i+1])  # Derivative of ReLU
            dW = np.dot(activations[i].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            
            # Update hidden layer weights
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db

    def fit(self, X):
        for epoch in range(self.epochs):
            # Forward pass
            activations = self.forward(X)
            
            # Compute loss
            loss = self.mse_loss(X, activations[-1])
            self.losses.append(loss)
            
            # Backward pass and weight update
            self.backward(X, activations)

    def transform(self, X):
        activations = self.forward(X)
        return activations[len(self.hidden_dims)]  # Return the encoded representation

    def reconstruct(self, X):
        activations = self.forward(X)
        return activations[-1]  # Return the reconstructed data

    def visualize(self, X_encoded, y):
        plt.figure()
        # Visualize the encoded data in 2D
        colors = ['r', 'b', 'g', 'y', 'c']
        markers = ['s', 'x', 'o', '^', 'D']
        for label, color, marker in zip(np.unique(y), colors, markers):
            plt.scatter(X_encoded[y == label, 0], X_encoded[y == label, 1], 
                        c=color, label=f'Class {label}', marker=marker)
        plt.xlabel('Encoded Feature 1')
        plt.ylabel('Encoded Feature 2')
        plt.title('Nonlinear Autoencoder - Encoded Data')
        plt.legend()
        plt.savefig('nonlinear_autoencoder.png')

    def plot_loss_curve(self):
        # Plot the loss curve
        plt.figure()
        plt.plot(range(self.epochs), self.losses, label='Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Nonlinear Autoencoder - Loss Curve')
        plt.legend()
        plt.savefig('nonlinear_autoencoder_loss.png')

def load_data():
    # Load and preprocess the wine dataset
    df_wine = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X = normalize_data(X)
    return X, y

def normalize_data(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_standardized = (X - mean) / std
    return X_standardized

def evaluate_model(model, X, y):
    # Evaluate the model using reconstruction error
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

    # Apply PCA and evaluate the model
    pca = PCA(n_components=2)
    pca.fit(X)
    evaluate_model(pca, X, y)
    pca.plot_explained_variance()

    # Train and evaluate the Linear Autoencoder
    linear_autoencoder = LinearAutoencoder(input_dim=X.shape[1], encoding_dim=6, learning_rate=0.005, epochs=30000)
    linear_autoencoder.fit(X)
    evaluate_model(linear_autoencoder, X, y)
    linear_autoencoder.plot_loss_curve()

    # Train and evaluate the Nonlinear Autoencoder
    nonlinear_autoencoder = NonlinearAutoencoder(input_dim=X.shape[1], hidden_dims=[16], output_dim=X.shape[1], learning_rate=0.007, epochs=30000)
    nonlinear_autoencoder.fit(X)
    evaluate_model(nonlinear_autoencoder, X, y)
    nonlinear_autoencoder.plot_loss_curve()
    
if __name__ == '__main__':
    main()
