import numpy as np
import matplotlib.pyplot as plt

class Perception:
    def __init__(self, n_feature=1, n_iter=200, lr=0.01, tol=None):
        self.n_iter = n_iter # number of iterations
        self.lr = lr # learning rate
        self.tol = tol # threshold for stopping iteration
        self.W = np.random.random(n_feature + 1) * 0.5 # weights
        self.loss = [] # loss
        self.best_loss = np.inf # best loss
        self.patience = 10 # patience for early stopping
    
    def _loss(self, y, y_pred):
        # loss function
        return -y_pred * y if y_pred * y < 0 else 0
    
    def _gradient(self, x_bar, y, y_pred):
        # gradient of loss function
        return -x_bar * y if y_pred * y < 0 else 0
    
    def _preprocess_data(self, X):
        # add bias term to X
        m, n = X.shape
        X_ = np.empty((m, n + 1))
        X_[:, 0] = 1
        X_[:, 1:] = X
        return X_
    
    def sgd_update(self, X, y):
        # stochastic gradient descent
        break_out = False
        epoch_no_improve = 0
        for i in range(self.n_iter):
            for i,x in enumerate(X):
                # predict the label of x
                y_pred = self._predict(x)
                # compute loss
                loss = self._loss(y[i], y_pred)
                self.loss.append(loss)
                if self.tol is not None:
                    if loss < self.best_loss - self.tol:
                        # update best loss
                        self.best_loss = loss
                        epoch_no_improve = 0
                    elif np.abs(loss - self.best_loss) < self.tol:
                        # no improvement
                        epoch_no_improve += 1
                        if epoch_no_improve >= self.patience:
                            print(f"Early stopping at epoch {i}")
                            break_out = True
                            break
                    else:
                        epoch_no_improve = 0
                # compute gradient and update weights
                grad = self._gradient(x, y[i], y_pred)
                self.W = self.W - self.lr * grad
            # check if break out
            if break_out:
                break_out = False
                break
            
    def _predict(self, X):
        # predict the label of x
        return X @ self.W
    
    def train(self, X_train, t_train):
        # train model
        X_train_bar = self._preprocess_data(X_train)
        print(X_train_bar)
        self.sgd_update(X_train_bar, t_train)
        print(self.W)
        
    def plot_loss(self):
        plt.plot(self.loss)
        plt.grid()
        plt.show()
        
if __name__ == "__main__":
    # generate data
    X_train = np.array([[-2,4],[4,1],[1,6],[2,4],[6,2]])
    y_train = np.array([-1,-1,1,1,1])
    # train model
    _, n_feature = X_train.shape
    model = Perception(n_feature=n_feature, n_iter=100, lr=0.01, tol=0.001)
    model.train(X_train, y_train)
    print(f"W: {model.W}")
    y_pred = np.sign(model._predict(model._preprocess_data(X_train)))
    print(f"y_pred: {y_pred}")
    # plot loss
    plt.figure()
    model.plot_loss()
    