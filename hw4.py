import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

def read_dataset():
    # read dataset
    column_names = [
        'Class Label', 'Alcohol', 'Malic Acid', 'Ash', 'Alcalinity of Ash', 
        'Magnesium', 'Total Phenols', 'Flavanoids', 'Nonflavanoid Phenols', 
        'Proanthocyanins', 'Color Intensity', 'Hue', 'OD280/OD315 of Diluted Wines', 
        'Proline'
    ]
    data = pd.read_csv('wine.data', header=None, names=column_names)
    # randomly remove a class
    unique_classes = data['Class Label'].unique()
    class_to_remove = np.random.choice(unique_classes)
    data_filtered = data[data['Class Label'] != class_to_remove]
    # shuffle the data
    data_filtered = data_filtered.sample(frac=1, random_state=42).reset_index(drop=True)
    print(data_filtered)
    return data_filtered

def mean_normalization(X):
    return (X - X.mean()) / X.std()

def split_data(data_filtered):
    # split data into features and labels
    data_filtered = data_filtered.sample(frac=1)
    # separate features and labels
    x = data_filtered.iloc[:, 1:]
    x = mean_normalization(x)
    y = data_filtered.iloc[:, 0]
    y = convert_labels(y)
    return x, y

def split_test_and_train(x, y):
    # split data into train and test
    x_train = x.iloc[:int(len(x)*0.7), :]
    x_test = x.iloc[int(len(x)*0.7):, :]
    y_train = y[:int(len(y)*0.7)]
    y_test = y[int(len(y)*0.7):]
    return x_train, x_test, y_train, y_test

def convert_labels(y):
    # convert labels to 0 and 1
    converted_labels = []
    y = np.array(y)
    for label in y:
        if label == y[0]:
            converted_labels.append(0)
        else:
            converted_labels.append(1)
    return np.array(converted_labels)

class LogisticRegression:
    def __init__(self, n_features=1, n_iter=200, lr=1e-3, alpha=0.01):
        self.n_iter = n_iter  # Maximum number of iterations
        self.lr = lr          # Learning rate
        self.W = np.random.random(n_features + 1) * 0.05  # Model parameters (weights)
        self.loss = []        # To store loss values during training
        self.test_loss = []   # To store test loss values during training
        self.alpha = alpha    # Regularization parameter
        self.val_loss = []    # To store validation loss values during training

    def _linear_tf(self, X):
        return X @ self.W  # Linear transformation of inputs

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))  # Sigmoid function for probability estimation

    def predict_probability(self, X):
        # Linear transformation and sigmoid activation function
        z = self._linear_tf(X)
        return self._sigmoid(z)

    def _loss(self, y, y_pred):
        epsilon = 1e-5  # Small constant to avoid log(0)
        loss = -np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))
        return loss

    def _gradient(self, X, y, y_pred):
        # add regularization term
        regularization_term = self.alpha * self.W
        return -(y - y_pred) @ X / y.size + regularization_term  # Gradient calculation

    def preprocess_data(self, X):
        m, n = X.shape
        X_new = np.empty((m, n + 1))  # Adding a bias term (intercept)
        X_new[:, 0] = 1  # Bias term (column of 1s)
        X_new[:, 1:] = X
        return X_new

    def k_fold_split(self, length, k):
        indices = np.arange(length)
        np.random.shuffle(indices)  # shuffle the indices
        folds = np.array_split(indices, k)  # split the data into k folds
        # yield the train and validation indices
        for i in range(k):
            train_indices = np.concatenate(folds[:i] + folds[i+1:])  # train set
            val_indices = folds[i]  # validation set
            yield train_indices, val_indices

    def batch_update(self, X, y, X_test, y_test, k=5, n_samples=20, n_iter_alpha=100):
        # find the best alpha
        best_alpha = None
        best_val_loss = float('inf')
        # Random search for alpha
        for _ in range(n_samples):
            loss1 = []
            test_loss1 = []
            current_alpha = np.random.uniform(0.001, 1.0)  # Randomly select alpha
            self.alpha = current_alpha  # Set current alpha
            # Train the model for a specified number of iterations
            for iter in range(n_iter_alpha):
                for train_indices, val_indices in self.k_fold_split(X.shape[0], k):
                    X_train, X_val = X[train_indices], X[val_indices]
                    y_train, y_val = y[train_indices], y[val_indices]
                    # Calculate training loss
                    y_pred = self.predict_probability(X_train)
                    loss = self._loss(y_train, y_pred)
                    loss1.append(loss)  # Store the loss value
                    # Calculate test loss
                    y_pred_test = self.predict_probability(X_test)
                    loss_test = self._loss(y_test, y_pred_test)
                    test_loss1.append(loss_test)  # Store the test loss value
                    # Update weights
                    grad = self._gradient(X_train, y_train, y_pred)
                    self.W = self.W - self.lr * grad  # Update weights
                # Calculate validation loss
                y_pred_val = self.predict_probability(X_val)
                val_loss = self._loss(y_val, y_pred_val)
            # Check if the current alpha has the best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_alpha = current_alpha
                self.loss = loss1  # Store the loss value
                self.test_loss = test_loss1  # Store the test loss value
                final_W = self.W  # Store the weights
        # Set the best alpha for further training
        self.alpha = best_alpha
        self.W = final_W
        print(f"Best alpha found: {best_alpha}, with validation loss: {best_val_loss}")

        # Continue training with the best alpha
        for iter in range(self.n_iter):
            indices = np.random.permutation(X.shape[0])
            X_train, y_train = X[indices], y[indices]
            # Calculate training loss
            y_pred = self.predict_probability(X_train)
            loss = self._loss(y_train, y_pred)
            self.loss.append(loss)  # Store training loss
            # Calculate test loss
            y_pred_test = self.predict_probability(X_test)
            loss_test = self._loss(y_test, y_pred_test)
            self.test_loss.append(loss_test)  # Store test loss
            # Update weights
            grad = self._gradient(X_train, y_train, y_pred)
            self.W = self.W - self.lr * grad  # Update weights

    def stochastic_update(self, X, y, X_test, y_test):        
        # Continue training with the best alpha
        for iter in range(self.n_iter):
            # shuffle the data
            indices = np.random.permutation(X.shape[0])
            for i in indices:
                # train loss
                y_pred = self.predict_probability(X[i])
                loss = self._loss(y[i], y_pred)
                self.loss.append(loss)  # Store the loss value
                # test loss
                y_pred_test = self.predict_probability(X_test)
                loss_test = self._loss(y_test, y_pred_test)
                self.test_loss.append(loss_test)  # Store the test loss value
                # update weights
                grad = self._gradient(X[i:i+1], np.array([y[i]]), np.array([y_pred]))
                self.W = self.W - self.lr * grad  # Update the weights

    def mini_batch_update(self, X, y, X_test, y_test, batch_size=64, k=5, n_samples=20, n_iter_alpha=100):
        # find the best alpha
        best_alpha = None
        best_val_loss = float('inf')
        # Random search for alpha
        for _ in range(n_samples):
            current_alpha = np.random.uniform(0.001, 1.0)  # Randomly select alpha
            self.alpha = current_alpha  # Set current alpha
            loss1 = []
            test_loss1 = []
            # Train the model for a specified number of iterations
            for iter in range(n_iter_alpha):
                # shuffle the data
                indices = np.random.permutation(X.shape[0])
                for i in range(0, X.shape[0], batch_size):
                    # get the batch
                    batch_indices = indices[i:min(i + batch_size, X.shape[0])]
                    X_batch, y_batch = X[batch_indices], y[batch_indices]
                    # split the data into k folds
                    for train_indices, val_indices in self.k_fold_split(X_batch.shape[0], k):
                        X_train, X_val = X_batch[train_indices], X_batch[val_indices]
                        y_train, y_val = y_batch[train_indices], y_batch[val_indices]
                        # Calculate training loss
                        y_pred = self.predict_probability(X_train)
                        loss = self._loss(y_train, y_pred)
                        loss1.append(loss)  # Store the loss value
                        # Calculate test loss
                        y_pred_test = self.predict_probability(X_test)
                        loss_test = self._loss(y_test, y_pred_test)
                        test_loss1.append(loss_test)  # Store the test loss value
                        # Update weights
                        grad = self._gradient(X_train, y_train, y_pred)
                        self.W = self.W - self.lr * grad  # Update the weights
                # Calculate validation loss
                y_pred_val = self.predict_probability(X_val)
                val_loss = self._loss(y_val, y_pred_val)
                # Check if the current alpha has the best validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_alpha = current_alpha
                    self.loss = loss1  # Store the loss value
                    self.test_loss = test_loss1  # Store the test loss value
                    final_W = self.W # Store the weights
        # Set the best alpha for further training
        self.alpha = best_alpha
        self.W = final_W
        print(f"Best alpha found: {best_alpha}, with validation loss: {best_val_loss}")
        
        # Continue training with the best alpha
        for iter in range(self.n_iter):
            # shuffle the data
            indices = np.random.permutation(X.shape[0])
            for i in range(0, X.shape[0], batch_size):
                # get the batch
                batch_indices = indices[i:min(i + batch_size, X.shape[0])]
                X_train, y_train = X[batch_indices], y[batch_indices]
                # train loss
                y_pred = self.predict_probability(X_train)
                loss = self._loss(y_train, y_pred)
                self.loss.append(loss)  # Store the loss value
                # test loss
                y_pred_test = self.predict_probability(X_test)
                loss_test = self._loss(y_test, y_pred_test)
                self.test_loss.append(loss_test)  # Store the test loss value
                # update weights
                grad = self._gradient(X_train, y_train, y_pred)
                self.W = self.W - self.lr * grad  # Update the weights

    def train(self, X_train, y_train, X_test, y_test, method='batch'):
        X_train = self.preprocess_data(X_train)
        X_test = self.preprocess_data(X_test)
        if method == 'batch':
            self.batch_update(X_train, y_train, X_test, y_test)
        elif method =='stochastic':
            self.stochastic_update(X_train, y_train, X_test, y_test)
        elif method =='mini-batch':
            self.mini_batch_update(X_train, y_train, X_test, y_test)
        else:
            raise ValueError("Invalid method. Choose 'batch','stochastic', or'mini-batch'.")

    def predict(self, X):
        X = self.preprocess_data(X)
        y_pred = self.predict_probability(X)
        return np.where(y_pred >= 0.5, 1, 0)  # Convert probabilities to class labels (0 or 1)
    
    def plot_loss(self):
        # plot loss
        plt.plot(self.loss, alpha=1, label='train loss')
        plt.plot(self.test_loss, alpha=1, label='test loss')
        # add title and labels
        plt.legend(['train loss', 'test loss'])
        plt.title('Loss over Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid()
        plt.savefig('loss.png')
        plt.show()
        print(f"Final train loss:{self.loss[-1]} test loss:{self.test_loss[-1]}")
        print(f"W:{self.W}, alpha:{self.alpha}")
    
    def evaluate(self, X, y):
        # predict the label of X
        y_pred = self.predict(X)
        y_pred = np.where(y_pred >= 0.5, 1, 0)
        # evaluate model accuracy, precision, recall, f1 score
        TP = sum((y_pred[i] == 1) and (y[i] == 1) for i in range(len(y)))
        FP = sum((y_pred[i] == 1) and (y[i] == 0) for i in range(len(y)))
        TN = sum((y_pred[i] == 0) and (y[i] == 0) for i in range(len(y)))
        FN = sum((y_pred[i] == 0) and (y[i] == 1) for i in range(len(y)))
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        print(f"Accuracy:{accuracy}, Precision:{precision}, Recall:{recall}, F1 score:{f1}")


def main():
    # read dataset
    data = read_dataset()
    x, y = split_data(data)
    X_train, X_test, y_train, y_test = split_test_and_train(x, y)
    
    # initialize the logistic regression model
    start_time = time.time()
    _, n_features = X_train.shape
    n_iter, lr = 10000, 0.005
    print(f"n of iteration:{n_iter}, loss rate:{lr}")
    model = LogisticRegression(n_features=n_features, n_iter=n_iter, lr=lr)

    # Train the model
    model.train(X_train, y_train, X_test, y_test, method='mini-batch')
    end_time = time.time()
    
    # plot loss and evaluate the model
    model.plot_loss()
    model.evaluate(x, y)
    print(f"Training time:{end_time - start_time}s")

if __name__ == "__main__":
    main()