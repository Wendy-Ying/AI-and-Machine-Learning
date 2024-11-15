import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Logistic Regression class (the code you provided)

class LogisticRegression:
    def __init__(self, n_features=1, n_iter=200, lr=1e-3, tol=None):
        self.n_iter = n_iter  # Maximum number of iterations
        self.lr = lr          # Learning rate
        self.tol = tol        # Tolerance for early stopping
        self.W = np.random.random(n_features + 1) * 0.05  # Model parameters (weights)
        self.loss = []        # To store loss values during training

    def _linear_tf(self, X):
        return X @ self.W  # Linear transformation of inputs

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))  # Sigmoid function for probability estimation

    def predict_probability(self, X):
        z = self._linear_tf(X)
        return self._sigmoid(z)

    def _loss(self, y, y_pred):
        epsilon = 1e-5  # Small constant to avoid log(0)
        loss = -np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))
        return loss

    def _gradient(self, X, y, y_pred):
        return -(y - y_pred) @ X / y.size  # Gradient calculation

    def preprocess_data(self, X):
        m, n = X.shape
        X_new = np.empty((m, n + 1))  # Adding a bias term (intercept)
        X_new[:, 0] = 1  # Bias term (column of 1s)
        X_new[:, 1:] = X
        return X_new

    def batch_update(self, X, y):
        if self.tol is not None:
            loss_old = np.inf  # Initialize the old loss for comparison

        for iter in range(self.n_iter):
            y_pred = self.predict_probability(X)
            loss = self._loss(y, y_pred)
            self.loss.append(loss)  # Store the loss value

            if self.tol is not None:
                if np.abs(loss_old - loss) < self.tol:  # Early stopping condition
                    break
                loss_old = loss

            grad = self._gradient(X, y, y_pred)
            self.W = self.W - self.lr * grad  # Update the weights

    def train(self, X_train, y_train):
        X_train = self.preprocess_data(X_train)
        self.batch_update(X_train, y_train)

    def predict(self, X):
        X = self.preprocess_data(X)
        y_pred = self.predict_probability(X)
        return np.where(y_pred >= 0.5, 1, 0)  # Convert probabilities to class labels (0 or 1)


# Main function to test the Logistic Regression class
def main():
    # Generate a synthetic binary classification dataset
    X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, random_state=42)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Feature scaling (optional but helps in gradient-based optimization)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize the logistic regression model
    model = LogisticRegression(n_features=2, n_iter=1000, lr=0.01)

    # Train the model
    model.train(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Evaluate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Plot the loss over iterations
    plt.plot(model.loss)
    plt.title('Loss Over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()


if __name__ == "__main__":
    main()
