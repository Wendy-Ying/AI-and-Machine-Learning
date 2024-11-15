import numpy as np

class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize the dimensions of the network
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases with random values
        self.weights1 = np.random.randn(self.input_size, self.hidden_size) # random values for weights
        self.bias1 = np.zeros((1, self.hidden_size))  # Biases for the hidden layer
        self.weights2 = np.random.randn(self.hidden_size, self.output_size) # random values for weights
        self.bias2 = np.zeros((1, self.output_size))  # Biases for the output layer

    def sigmoid(self, x):
        # Activation function: Sigmoid
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Derivative of the Sigmoid function
        return x * (1 - x)

    def forward(self, inputs):
        # Forward pass through the network
        self.hidden_layer = self.sigmoid(np.dot(inputs, self.weights1) + self.bias1)
        self.output_layer = self.sigmoid(np.dot(self.hidden_layer, self.weights2) + self.bias2)
        return self.output_layer

    def backward(self, inputs, targets, learning_rate):
        # Backpropagation process
        # Calculate the error at the output layer
        output_errors = targets - self.output_layer
        output_delta = output_errors * self.sigmoid_derivative(self.output_layer)

        # Calculate the error for the hidden layer
        hidden_errors = output_delta.dot(self.weights2.T)
        hidden_delta = hidden_errors * self.sigmoid_derivative(self.hidden_layer)

        # Update the weights and biases using the calculated deltas
        self.weights2 += self.hidden_layer.T.dot(output_delta) * learning_rate
        self.bias2 += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights1 += inputs.T.dot(hidden_delta) * learning_rate
        self.bias1 += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
        
    def train(self, inputs, targets, learning_rate, epochs):
        # Training the model over a specified number of epochs
        for epoch in range(epochs):
            # Perform a forward pass
            self.forward(inputs)
            # Perform a backward pass (weight updates)
            self.backward(inputs, targets, learning_rate)

    def predict(self, inputs):
        # Predict the output for given inputs
        return self.forward(inputs)

# Example usage
if __name__ == "__main__":
    # Create a simple dataset
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])  # XOR problem

    # Initialize the MLP
    mlp = MultiLayerPerceptron(input_size=2, hidden_size=3, output_size=1)

    # Train the model
    mlp.train(inputs, targets, learning_rate=0.1, epochs=10000)

    # Make predictions
    predictions = mlp.predict(inputs)
    print("Predictions after training:")
    print(predictions)
