import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from multi_layer_perceptron import MultiLayerPerceptron

def mean_normalization(df):
    normalized_df = (df - df.mean()) / (df.max() - df.min())
    return normalized_df

def load_data():
    # load data
    train_data = pd.read_csv('optdigits.tra', header=None)
    test_data = pd.read_csv('optdigits.tes', header=None)
    # split data into features and labels
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    # standardize data
    X_train = mean_normalization(X_train)
    X_test = mean_normalization(X_test)
    return X_train, y_train, X_test, y_test
    
def evaluate(model, X, y):
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    return accuracy

def plot_confusion_matrix(y_true, y_pred, num_classes=10):
    labels = [str(i) for i in range(num_classes)]
    # compute confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1
    # plot confusion matrix
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure()
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix.png')
    plt.show()

# load data
X_train, y_train, X_test, y_test = load_data()

# initialize MLP
mlp = MultiLayerPerceptron([64, 80, 32, 10], n_iter=240, lr=1e-3, batch_size=32)

# train model
mlp.train(X_train, y_train, X_test, y_test)

# draw weights and biases
mlp.plot_weights_and_biases()

# draw loss
mlp.plot_loss()

# evaluate model
test_accuracy = evaluate(mlp, X_test, y_test)
print(f'Test accuracy: {test_accuracy}')

# plot confusion matrix
y_test_pred = mlp.predict(X_test).astype(int)
plot_confusion_matrix(y_test, y_test_pred, num_classes=10)