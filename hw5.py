import numpy as np
import matplotlib.pyplot as plt

def generate_data_classifier(n_samples):
    # generate data for binary classification
    X = [] # input features
    y = [] # output labels
    
    for _ in range(n_samples):
        # 0 class data
        x0 = np.random.multivariate_normal(mean=[-1.8, -1.5], cov=[[0.3, 0], [0, 1.2]]) + np.random.normal(0, 0.1, 2)
        X.append(x0)
        y.append(0)
        # 1 class data
        x1 = np.random.multivariate_normal(mean=[1.2, 0.9], cov=[[1.3, 0], [0, 0.5]]) + np.random.normal(0, 0.1, 2)
        X.append(x1)
        y.append(1)
        x2 = np.random.multivariate_normal(mean=[2, 3], cov=[[0.7, 0], [0, 0.6]]) + np.random.normal(0, 0.1, 2)
        X.append(x2)
        y.append(0)
        # 1 class data
        x3 = np.random.multivariate_normal(mean=[1.5, -1.2], cov=[[0.4, 0], [0, 1.6]]) + np.random.normal(0, 0.1, 2)
        X.append(x3)
        y.append(1)
        x4 = np.random.multivariate_normal(mean=[3.4, -3.8], cov=[[0.9, 0], [0, 0.2]]) + np.random.normal(0, 0.1, 2)
        X.append(x4)
        y.append(0)
        # 1 class data
        x5 = np.random.multivariate_normal(mean=[-2.1, 1.9], cov=[[0.5, 0], [0, 0.7]]) + np.random.normal(0, 0.1, 2)
        X.append(x5)
        y.append(1)
    
    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    # visualize the data
    plt.figure()
    plt.scatter(X[y.ravel() == 0][:, 0], X[y.ravel() == 0][:, 1], color='red', label='Class 0')
    plt.scatter(X[y.ravel() == 1][:, 0], X[y.ravel() == 1][:, 1], color='green', label='Class 1')
    plt.title('Binary Classification Data Generated with NumPy')
    plt.legend()
    plt.savefig('classifier_dataset.png')
    # plt.show()
    plt.close()
    return X, y

def generate_nonlinear_dataset(n_samples=300):
    # generate input features
    X = np.linspace(-2 * np.pi, 3 * np.pi, n_samples).reshape(-1, 1)
    noise_level = 0.1  # Adjust this value to control the noise level
    # generate output labels
    y = np.sin(X) + np.random.normal(0, noise_level, X.shape)

    # visualize the data
    plt.figure()
    plt.scatter(X, y, color='blue', label='Data with Noise')
    plt.plot(X, np.sin(X), color='red', label='True Function', linewidth=2)
    plt.title('Nonlinear Dataset with Noise')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.legend()
    plt.savefig('nonlinear_dataset.png')
    # plt.show()
    plt.close()
    return X, y

class MultiLayerPerceptron:
    def __init__(self, layer_sizes, dataset_type):
        # layer_sizes is a list containing the number of nodes in each layer
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        self.dataset = dataset_type
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        self.target_max = np.max(targets)
        self.target_min = np.min(targets)
        
        # Initialize weights and biases for each layer
        for i in range(self.num_layers - 1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1])
            bias = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def activation(self, x):
        return 1 / (1 + np.exp(-x)) # Sigmoid activation function

    def activation_derivative(self, x):
        return x * (1 - x) # Derivative of the Sigmoid function

    def min_max_normalize(self, data):
        return (data - self.target_min) / (data.max() - self.target_min)

    def min_max_denormalize(self, data):
        return data * (self.target_max - self.target_min) + self.target_min

    def forward(self, inputs):
        # Forward pass through the network
        self.activations = [inputs]  # Store activations of each layer
        for i in range(self.num_layers - 1):
            # Compute the weighted sum and apply activation function
            z = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            activation = self.activation(z)
            self.activations.append(activation)
        return self.activations[-1]

    def backward(self, inputs, targets, learning_rate):
        # Backpropagation process
        output_errors = targets - self.activations[-1]
        output_delta = output_errors * self.activation_derivative(self.activations[-1])
        
        # Update weights and biases for the output layer
        self.weights[-1] += self.activations[-1].T.dot(output_delta) * learning_rate
        self.biases[-1] += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

        # Calculate errors and update weights for hidden layers
        hidden_errors = output_delta.dot(self.weights[-1].T) # initialize hidden_errors with output_delta for the last layer
        for i in range(self.num_layers - 3, -1, -1):
            # Calculate hidden delta for the current layer
            hidden_delta = hidden_errors * self.activation_derivative(self.activations[i + 1])
            
            # Update weights and biases for the current hidden layer
            self.weights[i] += self.activations[i].T.dot(hidden_delta) * learning_rate
            self.biases[i] += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
            
            # Update hidden_errors for the next layer's calculation
            hidden_errors = hidden_delta.dot(self.weights[i].T)

    def mean_squared_error(self, predictions, targets):
        return np.mean((predictions - targets) ** 2)

    def bgd_train(self, inputs, targets, learning_rate, epochs):
        # Train the model over specified epochs
        for epoch in range(epochs):
            # Shuffle the data
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
            inputs = inputs[indices]
            targets = targets[indices]
            # Process each input
            self.forward(inputs)
            self.backward(inputs, targets, learning_rate)

    def mbgd_train(self, inputs, targets, learning_rate, epochs, batch_size=32):
        # Train the model over specified epochs with mini-batch gradient descent
        for epoch in range(epochs):
            # Shuffle training data for each epoch to improve generalization
            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)
            inputs_train = inputs[indices]
            targets_train = targets[indices]

            # Process each mini-batch
            for start in range(0, len(inputs_train), batch_size):
                end = min(start + batch_size, len(inputs_train))
                batch_inputs = inputs_train[start:end]
                batch_targets = targets_train[start:end]
                
                # Forward pass and get predictions
                predictions = self.forward(batch_inputs)
                # Backward pass for the mini-batch
                self.backward(batch_inputs, batch_targets, learning_rate)
                
                # Calculate loss for the current mini-batch
                batch_loss = self.mean_squared_error(predictions, batch_targets)

    def sgd_train(self, inputs, targets, learning_rate, epochs):
        # Train the model over specified epochs using stochastic gradient descent
        for epoch in range(epochs):
            # Shuffle training data for each epoch to improve generalization
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
            inputs_train = inputs[indices]
            targets_train = targets[indices]
            
            # Iterate over each sample in the training set
            for i in range(len(inputs_train)):
                # Forward pass for a single sample
                predictions = self.forward(inputs_train[i:i+1])
                # Backward pass for the same sample
                self.backward(inputs_train[i:i+1], targets_train[i:i+1], learning_rate)
                
                # Calculate loss for the current sample
                batch_loss = self.mean_squared_error(predictions, targets_train[i:i+1])

    def k_fold_split(self, length, k):
        indices = np.arange(length)
        np.random.shuffle(indices)  # shuffle the indices
        folds = np.array_split(indices, k)  # split the data into k folds
        # yield the train and validation indices
        for i in range(k):
            train_indices = np.concatenate(folds[:i] + folds[i+1:])  # train set
            val_indices = folds[i]  # validation set
            yield train_indices, val_indices

    def train(self, inputs, targets, learning_rate, epochs, method='bgd'):
        # Train the model over specified epochs with mini-batch gradient descent
        if method == 'bgd':
            self.bgd_train(inputs, targets, learning_rate, epochs)
        elif method =='mbgd':
            self.mbgd_train(inputs, targets, learning_rate, epochs)
        elif method =='sgd':
            self.sgd_train(inputs, targets, learning_rate, epochs)
        else:
            print("Invalid method. Please choose 'bgd','mbgd', or'sgd'.")

    def predict(self, inputs):
        # Predict the output for given inputs
        return self.forward(inputs)

    def evaluate(self, inputs, targets):
        # predict the label of the input
        y_pred = self.predict(inputs)
        y_pred = np.array(y_pred)
        y_pred = np.where(y_pred >= 0.5, 1, 0) # convert the output to binary labels

        # evaluate model accuracy, precision, recall, f1 score
        TP = sum((y_pred[i][0] == 1) and (targets[i][0] == 1) for i in range(len(targets)))
        FP = sum((y_pred[i][0] == 1) and (targets[i][0] == 0) for i in range(len(targets)))
        TN = sum((y_pred[i][0] == 0) and (targets[i][0] == 0) for i in range(len(targets)))
        FN = sum((y_pred[i][0] == 0) and (targets[i][0] == 1) for i in range(len(targets)))
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return accuracy, precision, recall, f1
    
    def visualize_predictions_classifier(self, inputs, targets):
        plt.figure()
        # Create a grid of points to predict
        x_min, x_max = inputs[:, 0].min() - 1, inputs[:, 0].max() + 1
        y_min, y_max = inputs[:, 1].min() - 1, inputs[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 200))

        # Predict the output for each point in the grid
        grid_inputs = np.c_[xx.ravel(), yy.ravel()]
        grid_predictions = self.predict(grid_inputs)
        grid_predictions = grid_predictions.reshape(xx.shape)
        
        # Convert predictions to binary classes (0 or 1)
        grid_predictions = np.where(grid_predictions >= 0.5, 1, 0)

        # Plot the decision boundary
        plt.contourf(xx, yy, grid_predictions, alpha=0.5, levels=[-0.1, 0.5, 1.1], colors=['red', 'green'])
        
        # Plot the real data points
        plt.scatter(inputs[targets.ravel() == 0][:, 0], inputs[targets.ravel() == 0][:, 1], color='red', label='True Class 0', edgecolor='k')
        plt.scatter(inputs[targets.ravel() == 1][:, 0], inputs[targets.ravel() == 1][:, 1], color='green', label='True Class 1', edgecolor='k')

        # Custom legend for prediction colors
        red_patch = plt.Line2D([0], [0], marker='o', color='w', label='Class 0', markerfacecolor='red', markersize=10)
        green_patch = plt.Line2D([0], [0], marker='o', color='w', label='Class 1', markerfacecolor='green', markersize=10)

        # Title and legend
        plt.title('Model Predictions vs True Labels')
        plt.legend(handles=[red_patch, green_patch])
        plt.savefig('predicted_dataset_classifier.png')
        # plt.show()
        plt.close()

    def visualize_predictions_nonlinear(self, inputs, targets):
        plt.figure()
        # predict the label of the input
        y_pred = self.predict(inputs)
        y_pred = self.min_max_denormalize(y_pred)
        # plot the predictions against the true values
        plt.scatter(inputs, targets, color='blue', label='True Values', alpha=0.5)
        plt.scatter(inputs, y_pred, color='red', label='Predicted Values', alpha=0.5)
        x_range = np.linspace(inputs.min(), inputs.max(), 100)
        plt.plot(x_range, np.sin(x_range), color='green', label='True Function', linewidth=2)
        plt.title('Model Predictions vs True Values')
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.legend()
        plt.savefig('predictions_vs_true_values.png')
        # plt.show()
        plt.close()

    def get_neurons_for_layers(self, layers):
        prime_sequence = [3, 7, 13, 19, 25, 29, 37, 49] # the sequnece for neurons in each layer
        neurons_per_layer = []

        if layers <= 0:
            return neurons_per_layer

        for i in range(layers):
            if i < (layers + 1) // 2:  # increase
                neurons_per_layer.append(prime_sequence[i] * 3)
            else:  # decrease
                neurons_per_layer.append(prime_sequence[layers - i - 1] * 2)

        return neurons_per_layer


    def incremental_model_training_nonlinear(self, inputs, targets, initial_layers, max_layers, learning_rate, epochs, method='mbgd', k=5, tolerance=1e-3, patience=2):
        targets_normalized = self.min_max_normalize(targets)
        
        best_model = None
        best_score = float('inf')
        no_improve_count = 0
        
        # K-fold cross-validation
        for layers in range(initial_layers, max_layers + 1):
            layer_sizes = [1] + self.get_neurons_for_layers(layers) + [1]  # input layer + hidden layers + output layer
            
            fold_scores = []  # Store scores for each fold
            
            for train_indices, val_indices in self.k_fold_split(len(inputs), k):
                model = MultiLayerPerceptron(layer_sizes,dataset_type='nonlinear')
                
                # Train on training set
                model.train(inputs[train_indices], targets_normalized[train_indices], learning_rate=learning_rate, epochs=epochs, method=method)
                
                # Validate on validation set
                predictions = model.predict(inputs[val_indices])
                mse = model.mean_squared_error(predictions, targets_normalized[val_indices])
                fold_scores.append(mse)

                print(f"Layers: {layer_sizes}, Fold MSE: {mse}")
                model.visualize_predictions_nonlinear(inputs[val_indices], targets[val_indices])

            average_mse = np.mean(fold_scores)  # Average score across folds
            print(f"In this layer test: {layer_sizes}, Average MSE: {average_mse}")
            print("---------------------------------------------------------------------")

            # Early stopping mechanism
            if average_mse <= best_score:
                if average_mse < best_score - tolerance:
                    no_improve_count = 0  # Reset counter
                best_score = average_mse
                best_model = model
            else:
                no_improve_count += 1  # Increment counter if no significant improvement
                
            # Stop if no significant improvement for consecutive `patience` times
            if no_improve_count >= patience:
                print("Early stopping triggered")
                break
        
        # # plot loss trend of the best model
        # plt.figure()
        # plt.plot(best_loss, label='Loss Trend')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.title('Best Model Loss Trend')
        # plt.legend()
        # plt.savefig('best_model_loss_trend.png')
        # plt.close()
        
        return best_model

    def incremental_model_training_classifier(self, inputs, targets, initial_layers, max_layers, learning_rate, epochs, method='mbgd', k=5, tolerance=1e-4, patience=3):
        best_model = None
        best_score = float('-inf')
        no_improve_count = 0

        # K-fold cross-validation
        for layers in range(initial_layers, max_layers + 1):
            layer_sizes = [inputs.shape[1]] + self.get_neurons_for_layers(layers) + [1]
            
            fold_scores = []  # Store scores for each fold
            accuracy_scores = []
            precision_scores = []
            recall_scores = []
            f1_scores = []
            
            for train_indices, val_indices in self.k_fold_split(len(inputs), k):
                model = MultiLayerPerceptron(layer_sizes, dataset_type='classify')
                
                # Train on training set
                model.train(inputs[train_indices], targets[train_indices], learning_rate=learning_rate, epochs=epochs, method=method)
                
                # Validate on validation set
                accuracy, precision, recall, f1 = model.evaluate(inputs[val_indices], targets[val_indices])
                accuracy_scores.append(accuracy)
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)
                
                print(f"Layers: {layer_sizes}, Fold Accuracy: {accuracy:.3f}, Fold Precision: {precision:.3f}, Fold Recall: {recall:.3f}, Fold F1 score: {f1:.3f}")
                model.visualize_predictions_classifier(inputs[val_indices], targets[val_indices])
            
            average_accuracy = np.mean(accuracy_scores)
            average_precision = np.mean(precision_scores)
            average_recall = np.mean(recall_scores)
            average_f1 = np.mean(f1_scores)
            print(f"In this layer test: {layer_sizes}, Average Accuracy: {average_accuracy}, Average Precision: {average_precision}, Average Recall: {average_recall}, Average F1 score: {average_f1}")
            print("---------------------------------------------------------------------")

            # Early stopping mechanism
            if average_accuracy >= best_score:
                if average_accuracy - best_score > tolerance:
                    no_improve_count = 0  # Reset counter
                best_score = average_accuracy
                best_model = model
            else:
                no_improve_count += 1  # Increment counter if no significant improvement

            # Stop if no significant improvement for consecutive `patience` times
            if no_improve_count >= patience:
                print("Early stopping triggered")
                break
        
        # # plot loss trend of the best model
        # plt.figure()
        # plt.plot(best_loss, label='Loss Trend')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.title('Best Model Loss Trend')
        # plt.legend()
        # plt.savefig('best_model_loss_trend.png')
        # plt.close()
        
        return best_model

    def plot_weights_and_biases(self):
        plt.figure(figsize=(10, 10))
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


if __name__ == "__main__":
    dataset = 'nonlinear'  # 'classify' or 'nonlinear'
    
    if dataset == 'classify':
        inputs, targets = generate_data_classifier(n_samples=200)
        method = 'mbgd'
        learning_rate, epochs = 0.01, 1000
        print(f"Training Classifier, learning rate: {learning_rate}, epochs: {epochs}, method: {method}")
        initial_layers = 1
        max_layers = 10
        mlp = MultiLayerPerceptron(layer_sizes=[inputs.shape[1]], dataset_type='classify')
        best_model = mlp.incremental_model_training_classifier(inputs, targets, initial_layers, max_layers, learning_rate, epochs, method=method)
        
        # Evaluate the best model
        accuracy, precision, recall, f1 = best_model.evaluate(inputs, targets)
        print(f"Best Model - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 score: {f1}")
        best_model.visualize_predictions_classifier(inputs, targets)
        best_model.plot_weights_and_biases()

    elif dataset == 'nonlinear':
        inputs, targets = generate_nonlinear_dataset(n_samples=1000)
        learning_rate, epochs = 0.1, 10000
        method = 'mbgd'
        print(f"Training Nonlinear, learning rate: {learning_rate}, epochs: {epochs}, method: {method}")
        initial_layers = 1
        max_layers = 10
        mlp = MultiLayerPerceptron(layer_sizes=[1], dataset_type='nonlinear')
        best_model = mlp.incremental_model_training_nonlinear(inputs, targets, initial_layers, max_layers, learning_rate, epochs, method=method)
        
        # Visualize the predictions
        mse = best_model.mean_squared_error(best_model.predict(inputs), targets)
        print(f"Best Model - MSE: {mse}")
        best_model.visualize_predictions_nonlinear(inputs, targets)
        best_model.plot_weights_and_biases()