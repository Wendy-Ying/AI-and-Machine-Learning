import numpy as np
import matplotlib.pyplot as plt
import time

class LinearRegression:
    def __init__(self, n_iter=200, lr=1e-3, batch_size=32):
        self.n_iter = n_iter
        self.lr = lr
        self.batch_size = batch_size
        self.W = None
        self.train_loss = []
        self.test_loss = []
    
    def preprocess_data_X(self, X):
        # add bias term to X
        m, n = X.shape
        X_ = np.empty([m, n + 1])
        X_[:, 0] = 1
        X_[:, 1:] = X
        return X_

    def predict(self, X):
        # predict y
        # X = self.preprocess_data_X(X)
        return X @ self.W
    
    def min_max_normalization(self, x):
        # normalize the data by min-max normalization
        _min = np.min(x, axis=0)
        _max = np.max(x, axis=0)
        _range = _max - _min
        normalized = (x - _min) / _range
        return normalized, _min, _max

    def mean_normalization(self, x):
        # normalize the data by mean normalization
        mu = np.mean(x, axis=0)
        sigma = np.std(x, axis=0)
        normalized = (x - mu) / sigma
        return normalized, mu, sigma
    
    def normalize(self, x, method=None):
        # normalize the data
        if method =='min_max':
            return self.min_max_normalization(x)
        elif method =='mean':
            return self.mean_normalization(x)
        else:
            return x, None, None
    
    def inverse_min_max_weight(self, W, _min, _max):
        # inverse min-max normalization
        W[:1] -= W[1:] * _min / (_max - _min)
        W[1:] /= (_max - _min)
        return W

    def inverse_mean_weight(self, W, mu, sigma):
        # inverse mean normalization
        W[:1] -= W[1:] * mu / sigma
        W[1:] /= sigma
        return W
    
    def random_split(self, x, y, test_size=0.1, random_state=None):
        # shuffle the data
        if random_state is not None:
            np.random.seed(random_state)
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        # split the data
        split_index = int(len(x) * (1-test_size))
        train_indices, test_indices = indices[:split_index], indices[split_index:]
        xtrain, ytrain = x[train_indices], y[train_indices]
        xtest, ytest = x[test_indices], y[test_indices]
        return xtrain, ytrain, xtest, ytest
    
    def plot_loss(self):
        # plot loss
        plt.plot(self.train_loss, label='Train Loss')
        plt.plot(self.test_loss, label='Test Loss', linestyle='--')
        # set title and labels
        plt.title("Loss over iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.savefig('loss.png')
        plt.show()

    def calculate_loss(self, y, y_pred):
        # MSE loss
        N = y.size
        return np.sum((y_pred - y) ** 2) / (2 * N)

    def gradient(self, X, y, y_pred):
        # gradient
        return (X.T @ (y_pred - y)) / y.size
    
    def train_BGD(self, X_train, y_train, X_test, y_test):
        # BGD
        self.W = 10*np.random.rand(X_train.shape[1] + 1)
        X_train, X_test = self.preprocess_data_X(X_train), self.preprocess_data_X(X_test)
        for _ in range(self.n_iter):
            # compute loss
            y_pred = self.predict(X_train)
            train_loss = self.calculate_loss(y_train, y_pred)
            self.train_loss.append(train_loss)
            y_pred_test = self.predict(X_test)
            test_loss = self.calculate_loss(y_test, y_pred_test)
            self.test_loss.append(test_loss)
            # gradient descent
            self.W -= self.lr * self.gradient(X_train, y_train, y_pred)

    def train_SGD(self, X_train, y_train, X_test, y_test):
        # SGD
        self.W = 10*np.random.rand(X_train.shape[1] + 1)
        X_train, X_test = self.preprocess_data_X(X_train), self.preprocess_data_X(X_test)
        N = y_train.size
        for _ in range(self.n_iter):
            # shuffle the data
            indices = np.random.permutation(N)
            for i in indices:
                x_i = X_train[i:i+1]
                y_i = y_train[i:i+1]
                # compute loss
                y_pred = self.predict(x_i)
                train_loss = self.calculate_loss(y_i, y_pred)
                self.train_loss.append(train_loss)
                y_pred_test = self.predict(X_test)
                test_loss = self.calculate_loss(y_test, y_pred_test)
                self.test_loss.append(test_loss)
                # compute gradient
                grad = self.gradient(x_i, y_i, y_pred)
                self.W -= self.lr * grad


    def train_MBGD(self, X_train, y_train, X_test, y_test):
        # MBGD
        self.W = 10*np.random.rand(X_train.shape[1] + 1)
        X_train, X_test = self.preprocess_data_X(X_train), self.preprocess_data_X(X_test)
        N = y_train.size
        for _ in range(self.n_iter):
            # shuffle the data
            indices = np.random.permutation(N)
            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)  # not out of range
                # get batch data
                batch_indices = indices[start:end]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]
                # predict and compute loss
                y_pred = self.predict(X_batch)
                train_loss = self.calculate_loss(y_batch, y_pred)
                self.train_loss.append(train_loss)
                y_pred_test = self.predict(X_test)
                test_loss = self.calculate_loss(y_test, y_pred_test)
                self.test_loss.append(test_loss)
                # compute gradient
                grad = self.gradient(X_batch, y_batch, y_pred)
                self.W -= self.lr * grad

    def train(self, X_train, y_train, X_test, y_test, method='BGD'):
        if method == 'BGD':
            self.train_BGD(X_train, y_train, X_test, y_test)
        elif method == 'SGD':
            self.train_SGD(X_train, y_train, X_test, y_test)
        elif method == 'MBGD':
            self.train_MBGD(X_train, y_train, X_test, y_test)
        else:
            raise ValueError("Invalid method. Choose from 'BGD', 'SGD', or 'MBGD'.")
        
    def plot_fit(self, X, y):
        # plot the input data
        plt.scatter(X, y, label='Data points', color='blue', alpha=0.5)
        # plot the fitted line
        y_fit = self.predict(self.preprocess_data_X(X))
        plt.plot(X, y_fit, label='Fitted line', color='red')
        # set title and labels
        plt.title("Linear Regression Fit")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.legend()
        plt.grid()
        plt.savefig('fit.png')
        plt.show()

if __name__ == "__main__":
    # record the time
    start_time = time.time()
    
    # generate data
    X = np.arange(100).reshape(100,1)
    a, b = 1, 10
    y = a * X + b + np.random.normal(0, 5, size=X.shape)
    y = y.reshape(-1)
    
    # set parameters
    n_iter, lr = 30000, 5e-4
    model = LinearRegression(n_iter=n_iter, lr=lr)
    
    # normalize the data
    method = 'min_max'
    X_normalized, param1, param2 = model.normalize(X, method=method)

    # split the data
    X_train, y_train, X_test, y_test = model.random_split(X_normalized, y, test_size=0.2)

    # train the model
    model.train(X_train, y_train, X_test, y_test, method='MBGD')
    model.plot_loss()
    end_time = time.time()

    # compute loss
    train_loss = model.calculate_loss(y_train, model.predict(model.preprocess_data_X(X_train)))
    test_loss = model.calculate_loss(y_test, model.predict(model.preprocess_data_X(X_test)))
    
    # inverse normalization
    if method == 'min_max':
        W_original = model.inverse_min_max_weight(model.W, param1, param2)
    elif method == 'mean':
        W_original = model.inverse_mean_weight(model.W, param1, param2)
    else:
        W_original = model.W
    
    # output the result
    model.plot_fit(X, y)
    print(f'n of iteration: {n_iter}, loss rate: {lr}')
    print(f'Learned weights: {W_original}, Training loss: {train_loss}, Testing loss: {test_loss}')
    print(f'Time: {end_time - start_time}s')
    