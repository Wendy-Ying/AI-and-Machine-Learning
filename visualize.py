import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('optdigits.tra', header=None)

X = train_data.iloc[:, :-1].values
y = train_data.iloc[:, -1].values

def visualize_samples(X, y, sample_count=10):
    plt.figure(figsize=(10, 4))
    for i in range(sample_count):
        plt.subplot(3, 10, i + 1)
        plt.imshow(X[i].reshape(8, 8), cmap='gray')
        plt.title(f"Label: {y[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('data.png')
    plt.show()

visualize_samples(X, y, sample_count=30)
