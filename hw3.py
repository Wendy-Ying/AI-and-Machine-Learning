import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    print(data_filtered)
    return data_filtered

def split_data(data_filtered):
    # split data into features and labels
    data_filtered = data_filtered.sample(frac=1)
    # separate features and labels
    x = data_filtered.iloc[:, 1:]
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
    # convert labels to -1 and 1
    converted_labels = []
    y = np.array(y)
    for label in y:
        if label == y[0]:
            converted_labels.append(-1)
        else:
            converted_labels.append(1)
    return np.array(converted_labels)


class Perceptron:
    def __init__(self, n_feature=1, n_iter=200, lr=0.01, tol=None):
        self.n_iter = n_iter # number of iterations
        self.lr = lr # learning rate
        self.tol = tol # threshold for stopping iteration
        self.W = np.random.random(n_feature + 1) * 0.5 # weights
        self.loss = [] # loss
        self.test_loss = [] # test loss
        self.best_loss = np.inf # best loss
        self.patience = 100 # patience for early stopping
    
    def _loss(self, y, y_pred):
        # loss function for single input
        return np.maximum(0, -y * y_pred)
    
    def _loss_batch(self, y, y_pred):
        # loss function for batch input
        loss_matrix = np.maximum(0, -y * y_pred)
        return loss_matrix.mean()
    
    def _gradient(self, x, y, y_pred):
        # gradient of loss function for single input
        return -x * y if y_pred * y < 0 else 0
    
    def _gradient_batch(self, X, y, y_pred):
        # gradient of loss function for batch input
        gradients = np.zeros_like(self.W)
        for i in range(X.shape[0]):
            if y_pred[i] * y[i] < 0:  # misclassified
                for j in range(X.shape[1]):
                    gradients[j] += -X[i][j] * y[i]
        return gradients / X.shape[0]
    
    def _preprocess_data(self, X):
        # add bias term to X
        m, n = X.shape
        X_ = np.empty((m, n + 1))
        X_[:, 0] = 1
        X_[:, 1:] = X
        return X_
    
    def sgd_update(self, X, y, X_test, y_test):
        # stochastic gradient descent
        break_out = False
        epoch_no_improve = 0
        for n in range(self.n_iter):
            for i,x in enumerate(X):
                # predict the label of x
                y_pred = self._predict(x)
                # print(f"Predicted label:{y_pred}, Actual label:{y[i]}")
                # compute loss
                loss = self._loss(y[i], y_pred)
                self.loss.append(loss)
                # compute test loss
                test_loss = self._loss_batch(y_test, self._predict(X_test))
                self.test_loss.append(test_loss)
                # check if break out
                if self.tol is not None:
                    if loss < self.best_loss - self.tol:
                        # update best loss
                        self.best_loss = loss
                        epoch_no_improve = 0
                    elif np.abs(loss - self.best_loss) < self.tol:
                        # no improvement
                        epoch_no_improve += 1
                        if epoch_no_improve >= self.patience:
                            print(f"Early stopping at epoch {n}")
                            break_out = True
                            break
                    else:
                        epoch_no_improve = 0
                # compute gradient and update weights
                grad = self._gradient(x, y[i], y_pred)
                self.W = self.W - self.lr * grad
            # print(f"Epoch:{n}, Train Loss:{loss}, Test Loss:{test_loss}")
            # check if break out
            if break_out:
                break_out = False
                break
    
    def batch_update(self, X, y, X_test, y_test):
        # batch gradient descent
        break_out = False
        epoch_no_improve = 0
        for n in range(self.n_iter):
            # predict the label of x
            y_pred = self._predict(X)
            # compute loss
            loss = self._loss_batch(y, y_pred)
            self.loss.append(loss)
            # compute test loss
            test_loss = self._loss_batch(y_test, self._predict(X_test))
            self.test_loss.append(test_loss)
            # check if break out
            if self.tol is not None:
                if loss < self.best_loss - self.tol:
                    # update best loss
                    self.best_loss = loss
                    epoch_no_improve = 0
                elif np.abs(loss - self.best_loss) < self.tol:
                    # no improvement
                    epoch_no_improve += 1
                    if epoch_no_improve >= self.patience:
                        print(f"Early stopping at epoch {n}")
                        break_out = True
                        break
                else:
                    epoch_no_improve = 0
            # compute gradient and update weights
            grad = self._gradient_batch(X, y, y_pred)
            self.W = self.W - self.lr * grad
            # print(f"Epoch:{n}, Train Loss:{loss}, Test Loss:{test_loss}")
            # check if break out
            if break_out:
                break_out = False
                break
            
    def _predict(self, X):
        # predict the label of x
        return X @ self.W
    
    def train(self, X_train, t_train, X_test, y_test, method='bgd'):
        # train model
        X_train_bar = self._preprocess_data(X_train)
        X_test_bar = self._preprocess_data(X_test)
        if method == 'bgd':
            self.batch_update(X_train_bar, t_train, X_test_bar, y_test)
        elif method =='sgd':
            self.sgd_update(X_train_bar, t_train, X_test_bar, y_test)
        else:
            raise ValueError('Invalid method')
        print(f"Weights:{self.W}")
        
    def plot_loss(self):
        # plot loss
        plt.plot(self.loss)
        plt.plot(self.test_loss)
        # add title and labels
        plt.legend(['train loss', 'test loss'])
        plt.title('Loss over Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid()
        plt.savefig('loss.png')
        plt.show()
        print(f"Final train loss:{self.best_loss} test loss:{self.test_loss[-1]}")
    
    def evaluate(self, X, y):
        # predict the label of X
        y_pred = self._predict(self._preprocess_data(X))
        y_pred = np.where(y_pred > 0, 1, -1)
        # evaluate model accuracy, precision, recall, f1 score
        TP = sum((y_pred[i] == 1) and (y[i] == 1) for i in range(len(y)))
        FP = sum((y_pred[i] == 1) and (y[i] == -1) for i in range(len(y)))
        TN = sum((y_pred[i] == -1) and (y[i] == -1) for i in range(len(y)))
        FN = sum((y_pred[i] == -1) and (y[i] == 1) for i in range(len(y)))
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        print(f"Accuracy:{accuracy}, Precision:{precision}, Recall:{recall}, F1:{f1}")
    
if __name__ == '__main__':
    # read dataset
    data = read_dataset()
    x, y = split_data(data)
    x_train, x_test, y_train, y_test = split_test_and_train(x, y)
    
    # initialize model
    _, n_feature = x_train.shape
    # n_iter, lr, tol = 100000, 6e-6, 1e-6 # bgd best
    n_iter, lr, tol = 10000, 1.5e-5, 1e-9 # sgd best
    print(f"n_iter:{n_iter}, lr:{lr}, tol:{tol}")
    model = Perceptron(n_feature=n_feature, n_iter=n_iter, lr=lr, tol=tol)
    
    # train model
    model.train(x_train, y_train, x_test, y_test, method='sgd')

    model.plot_loss()
    
    # evaluate model
    model.evaluate(x, y)