import numpy as np
import decision_tree as DT
from decisiontreeplotter import DecisionTreePlotter

# Reload your decision tree module if needed
import importlib
importlib.reload(DT)

def train_test_split(X, y, test_size=0.1, random_state=None):
    assert X.shape[0] == y.shape[0], "X and y should have the same number of samples"
    if random_state is not None:
        np.random.seed(random_state)

    total_samples = X.shape[0]
    test_samples = int(total_samples * test_size)
    # Randomly shuffle the indices
    indices = np.random.permutation(total_samples)
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]
    # Split the data into training and test sets
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

# Load the dataset
data = np.loadtxt('../dataset/lenses/lenses.data', dtype=int)

# Features and labels
X = data[:, 1:-1]
y = data[:, -1]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Create and train the decision tree
dt01 = DT.DecisionTree()
dt01.train(X_train, y_train)

# Print the tree
print(dt01.tree_)

# Use the trained tree to make predictions on the test set
y_pred = dt01.predict(X_test)

# Calculate accuracy
accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f"Accuracy: {accuracy}")

# Optionally, you can visualize the decision tree
features_dict = {
    0: {'name': 'age', 'value_names': {1:'young', 2:'pre-presbyopic', 3:'presbyopic'}},
    1: {'name':'prescript', 'value_names': {1: 'myope', 2: 'hypermetrope'}},
    2: {'name': 'astigmatic', 'value_names': {1: 'no', 2: 'yes'}},
    3: {'name': 'tear rate', 'value_names': {1:'reduced', 2:'normal'}},
}

label_dict = {
    1: 'hard',
    2: 'soft',
    3: 'no_lenses',
}

dtp = DecisionTreePlotter(dt01.tree_, feature_names=features_dict, label_names=label_dict)
dtp.plot()