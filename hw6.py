from scipy.spatial import KDTree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_dataset():
    # read dataset
    column_names = [
        'ID number', 'Diagnosis', 
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean','smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave_points_mean','symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se','smoothness_se',
        'compactness_se', 'concavity_se', 'concave_points_se','symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst','smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave_points_worst','symmetry_worst', 'fractal_dimension_worst',
    ]
    data = pd.read_csv('wdbc.data', header=None, names=column_names)
    return split_data(data)

def split_data(data):
    # split data into features and labels
    data = data.sample(frac=1)
    # separate features and labels
    x = data.iloc[:, 2:]
    y = data.iloc[:, 1]
    return split_test_and_train(x, y)

def split_test_and_train(x, y):
    # split data into train and test
    x = x.values
    x = normalize_data(x)
    x_train = x[:int(len(x)*0.7), :]
    x_test = x[int(len(x)*0.7):, :]
    y = y.values
    y_train = y[:int(len(y)*0.7)]
    y_test = y[int(len(y)*0.7):]
    return x_train, x_test, y_train, y_test

def normalize_data(x):
    # normalize data
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    return x

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.kdtree = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.kdtree = KDTree(X)

    def predict_multiple(self, X):
        # predict the label of multiple points
        predictions = [self.predict_single(x) for x in X]
        return np.array(predictions)

    def predict_single(self, x):
        # predict the label of a single point
        dist, idx = self.kdtree.query(x, k=self.k, p=2) # Find k nearest neighbors
        if idx.ndim == 1:
            idx = [idx] # Convert to a list if it's a 1D array
        if self.k == 1:
            neighbors_labels = [self.y_train[idx]]  # Just take the label of the single nearest neighbor
        else:
            neighbors_labels = [self.y_train[i] for i in idx[0]]  # Get labels of neighbors
        prediction = max(set(neighbors_labels), key=neighbors_labels.count) # Majority vote
        return prediction
    
    def evaluate(self, X_test, y_test):
        # predict the label of multiple points
        predictions = self.predict_multiple(X_test)
        # calculate TP, TN, FP, FN
        TP = np.sum((predictions == 'M') & (y_test == 'M'))
        TN = np.sum((predictions == 'B') & (y_test == 'B'))
        FP = np.sum((predictions == 'M') & (y_test == 'B'))
        FN = np.sum((predictions == 'B') & (y_test == 'M'))
        # calculate accuracy, precision, recall, f1
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        return accuracy, precision, recall, f1

def plot_evaluation_metrics(accuracy_list, precision_list, recall_list, f1_list):
    # plot the evaluation metrics
    plt.figure()
    plt.plot(range(1, 11), accuracy_list, label='Accuracy')
    plt.plot(range(1, 11), precision_list, label='Precision')
    plt.plot(range(1, 11), recall_list, label='Recall')
    plt.plot(range(1, 11), f1_list, label='F1')
    plt.xlabel('k')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.title('Evaluation Metrics')
    plt.legend()
    plt.grid()
    plt.savefig('Evaluation Metrics.png')
    plt.show()
    plt.close()
    
if __name__ == "__main__":
    # read dataset
    x_train, x_test, y_train, y_test = read_dataset()
    
    # test different k values
    accuracy_list, precision_list, recall_list, f1_list = [], [], [], []
    for k in range(1, 11):
        # Create KNN instance and fit the model
        knn = KNN(k=k)
        knn.fit(x_train, y_train)
        # Evaluate the model
        accuracy, precision, recall, f1 = knn.evaluate(x_test, y_test)
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        print(f"k: {k},\tAccuracy: {accuracy:.16f},\tPrecision: {precision:.16f},\tRecall: {recall:.16f},\tF1: {f1:.16f}")
    
    # plot the evaluation metrics
    plot_evaluation_metrics(accuracy_list, precision_list, recall_list, f1_list)