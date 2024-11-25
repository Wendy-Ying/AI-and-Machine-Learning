import numpy as np
import matplotlib.pyplot as plt

class MultiLayerPerceptron:
    def __init__(self, layer_sizes, n_iter=200, lr=1e-3, batch_size=128, dropout_rate=0.2, l2_lambda=3e-4):
        self.layer_sizes = layer_sizes  # Number of neurons in each layer
        self.n_iter = n_iter  # Number of iterations (epochs)
        self.lr = lr  # Initial learning rate
        self.batch_size = batch_size  # Batch size for training
        self.dropout_rate = dropout_rate  # Dropout rate
        self.l2_lambda = l2_lambda  # L2 regularization strength
        self.num_layers = len(layer_sizes)  # Total number of layers
        self.training = False  # Training mode

        # Initialize weights and biases with random values
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i]) 
                        for i in range(self.num_layers - 1)]
        self.biases = [np.zeros((1, layer_sizes[i + 1])) for i in range(self.num_layers - 1)]

        # Adam optimizer parameters
        self.m_weights = [np.zeros_like(w) for w in self.weights]
        self.v_weights = [np.zeros_like(w) for w in self.weights]
        self.m_biases = [np.zeros_like(b) for b in self.biases]
        self.v_biases = [np.zeros_like(b) for b in self.biases]
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

    def activation(self, x):
        # relu
        return np.maximum(0, x)

    def activation_derivative(self, x):
        # Derivative of leaky relu
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        # Softmax activation for the output layer
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability improvement
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def dropout(self, x):
        if self.training:
            mask = np.random.rand(*x.shape) > self.dropout_rate
            return x * mask / (1 - self.dropout_rate)
        return x

    def forward(self, inputs):
        # Forward pass with Dropout for hidden layers
        self.z_values = []
        self.activations = [inputs]
        for i in range(self.num_layers - 2):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            activation = self.activation(z)
            activation = self.dropout(activation)  # Apply dropout
            self.activations.append(activation)
        # Output layer with Softmax
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        activation = self.softmax(z)
        self.activations.append(activation)
        return self.activations[-1]

    def compute_loss(self, predictions, targets):
        # Cross-entropy loss with L2 regularization
        epsilon = 1e-10
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        weights = np.where(targets == 1, 1.2, 1.0)
        cross_entropy_loss = -np.mean(np.sum(weights * targets * np.log(predictions), axis=1))
        l2_loss = self.l2_lambda * sum(np.sum(w ** 2) for w in self.weights)
        return cross_entropy_loss + l2_loss
    
    def one_hot_encode(self, targets, num_classes):
        return np.eye(num_classes)[targets]

    def backward(self, inputs, targets, epoch):
        m = inputs.shape[0]
        predictions = self.activations[-1]
        delta = predictions - targets

        # Update output layer
        grad_w = np.dot(self.activations[-2].T, delta) / m + 2 * self.l2_lambda * self.weights[-1]
        grad_b = np.sum(delta, axis=0, keepdims=True) / m
        self.update_params(-1, grad_w, grad_b, epoch)

        # Backpropagate through hidden layers
        for i in range(self.num_layers - 2, 0, -1):
            delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(self.z_values[i - 1])
            grad_w = np.dot(self.activations[i - 1].T, delta) / m + 2 * self.l2_lambda * self.weights[i - 1]
            grad_b = np.sum(delta, axis=0, keepdims=True) / m
            self.update_params(i - 1, grad_w, grad_b, epoch)

    def update_params(self, layer_index, grad_w, grad_b, epoch):
        # Adam optimizer update rule
        self.m_weights[layer_index] = self.beta1 * self.m_weights[layer_index] + (1 - self.beta1) * grad_w
        self.v_weights[layer_index] = self.beta2 * self.v_weights[layer_index] + (1 - self.beta2) * (grad_w ** 2)
        m_hat_w = self.m_weights[layer_index] / (1 - self.beta1 ** (epoch + 1))
        v_hat_w = self.v_weights[layer_index] / (1 - self.beta2 ** (epoch + 1))

        self.m_biases[layer_index] = self.beta1 * self.m_biases[layer_index] + (1 - self.beta1) * grad_b
        self.v_biases[layer_index] = self.beta2 * self.v_biases[layer_index] + (1 - self.beta2) * (grad_b ** 2)
        m_hat_b = self.m_biases[layer_index] / (1 - self.beta1 ** (epoch + 1))
        v_hat_b = self.v_biases[layer_index] / (1 - self.beta2 ** (epoch + 1))

        self.weights[layer_index] -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
        self.biases[layer_index] -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

    def train(self, inputs, targets, test_inputs, test_targets):
        self.train_loss_history = []
        self.test_loss_history = []
        
        targets = self.one_hot_encode(targets, self.layer_sizes[-1])
        test_targets = self.one_hot_encode(test_targets, self.layer_sizes[-1])

        for epoch in range(self.n_iter):
            # Shuffle the training data
            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)
            shuffled_inputs = inputs[indices]
            shuffled_targets = targets[indices]

            # Process data in batches
            for start in range(0, len(shuffled_inputs), self.batch_size):
                end = min(start + self.batch_size, len(shuffled_inputs))
                batch_inputs = shuffled_inputs[start:end]
                batch_targets = shuffled_targets[start:end]
                self.forward(batch_inputs)
                self.backward(batch_inputs, batch_targets, epoch)

            # Record training and testing loss
            train_predictions = self.forward(inputs)
            test_predictions = self.forward(test_inputs)
            train_loss = self.compute_loss(train_predictions, targets)
            test_loss = self.compute_loss(test_predictions, test_targets)
            self.train_loss_history.append(train_loss)
            self.test_loss_history.append(test_loss)
            
            # Print progress
            if (epoch + 1) % 40 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{self.n_iter}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
            
            self.lr = 0.999 * self.lr
            
    def predict(self, inputs):
        # Forward pass to get softmax probabilities
        probabilities = self.forward(inputs)
        # Return the class index with the highest probability
        return np.argmax(probabilities, axis=1)

    def plot_loss(self):
        plt.plot(self.train_loss_history, label="Train Loss")
        plt.plot(self.test_loss_history, label="Test Loss")
        plt.legend()
        plt.title("Loss Over Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.savefig("mlp_loss.png")
        plt.show()
        
    def plot_weights_and_biases(self):
        plt.figure()
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            plt.subplot(len(self.weights), 1, i + 1)
            
            # plot weight distribution
            plt.hist(weight.flatten(), bins=30, alpha=0.7, label=f'Layer {i + 1} Weights', color='blue')
            
            # plot bias distribution
            plt.hist(bias.flatten(), bins=30, alpha=0.7, label=f'Layer {i + 1} Biases', color='orange')
            
            plt.title(f'Weight and Bias Distribution for Layer {i + 1}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.legend()
            
        plt.savefig('weights_and_biases_distribution.png')
        plt.close()