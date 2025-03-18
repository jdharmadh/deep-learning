import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

alpha = 0.3

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(a):
    return a * (1 - a)
def relu(x):
    return np.maximum(0, x)
def relu_derivative(a):
    return np.where(a > 0, 1, 0)
def activate_hidden(x):
    return relu(x)
def derivative_hidden(a):
    return relu_derivative(a)
def activate_output(x):
    return sigmoid(x)
def derivative_output(a):
    return sigmoid_derivative(a)

class Neuron:
    def __init__(self, num_weights, output=False):
        self.weights = np.random.randn(num_weights) * 0.1
        self.bias = np.random.randn() * 0.1
        self.last_activation = None
        self.last_input = None
        self.output = output

    def forward(self, x):
        self.last_input = x
        # Support both single sample (1D) and batch (2D: samples x features)
        if x.ndim == 1:
            z = np.dot(self.weights, x) + self.bias
        else:
            z = np.dot(x, self.weights) + self.bias
        if self.output:
            self.last_activation = activate_output(z)
        else:
            self.last_activation = activate_hidden(z)
        return self.last_activation

    def backward(self, error_from_next_layer):
        global alpha
        # Single sample mode
        if self.last_activation.ndim == 0 or self.last_activation.ndim == 1 and self.last_input.ndim == 1:
            if self.output:
                layer_error = error_from_next_layer * derivative_output(self.last_activation)
            else:
                layer_error = error_from_next_layer * derivative_hidden(self.last_activation)
            weights_grad = layer_error * self.last_input
            bias_grad = layer_error
            error_to_return = self.weights * layer_error
            self.weights -= alpha * weights_grad
            self.bias -= alpha * bias_grad
            return error_to_return
        else:
            # Batch mode: last_activation shape (batch_size, )
            if self.output:
                d = error_from_next_layer * derivative_output(self.last_activation)
            else:
                d = error_from_next_layer * derivative_hidden(self.last_activation)
            batch_size = self.last_activation.shape[0]
            # Gradients: average over the batch
            weights_grad = np.dot(self.last_input.T, d) / batch_size
            bias_grad = np.mean(d)
            # Propagate the error: for each sample, error = d * weights (element–wise multiplication per sample)
            error_to_return = d[:, np.newaxis] * self.weights  # shape: (batch_size, num_inputs)
            self.weights -= alpha * weights_grad
            self.bias -= alpha * bias_grad
            return error_to_return

class Layer:
    def __init__(self, previous_layer_neurons, num_neurons, output=False):
        self.previous_layer_neurons = previous_layer_neurons
        self.num_neurons = num_neurons
        self.neurons = [Neuron(self.previous_layer_neurons, output=output) for _ in range(num_neurons)]
        self.last_input = None
        self.last_output = None

    def forward(self, x):
        # In batch mode, x shape: (batch_size, previous_layer_neurons)
        self.last_input = x
        # Compute each neuron’s output.
        outputs = [n.forward(x) for n in self.neurons]
        # If batch, each output is (batch_size,) so stack as columns.
        if x.ndim > 1:
            self.last_output = np.column_stack(outputs)  # shape: (batch, num_neurons)
        else:
            self.last_output = np.array(outputs)  # single sample
        return self.last_output

    def backward(self, error_from_next_layer):
        # For a single sample: error_from_next_layer shape: (num_neurons,)
        if error_from_next_layer.ndim == 1:
            errors_to_return = np.zeros(self.previous_layer_neurons)
            for i, neuron in enumerate(self.neurons):
                neuron_error = neuron.backward(error_from_next_layer[i])
                errors_to_return += neuron_error
            return errors_to_return
        else:
            # Batch mode: error_from_next_layer shape: (batch_size, num_neurons)
            batch_size = error_from_next_layer.shape[0]
            errors_to_return = np.zeros((batch_size, self.previous_layer_neurons))
            for i, neuron in enumerate(self.neurons):
                # Pass the error column for the i-th neuron.
                neuron_error = neuron.backward(error_from_next_layer[:, i])
                errors_to_return += neuron_error
            return errors_to_return

class NeuralNetwork():
    def __init__(self, input_features, num_layers, neurons_per_layer, output_features):
        self.input_features = input_features
        self.num_layers = num_layers
        self.output_features = output_features
        self.layers = []
        self.layers.append(Layer(input_features, neurons_per_layer))  # Input layer
        for _ in range(num_layers - 2):
            self.layers.append(Layer(neurons_per_layer, neurons_per_layer))
        self.layers.append(Layer(neurons_per_layer, output_features, output=True))  # Output layer

    def sgd(self, x, y):
        # Single–sample SGD, unchanged.
        output = x
        for l in self.layers:
            output = l.forward(output)
        error = output - y
        for l in reversed(self.layers):
            error = l.backward(error)
        return (output - y) ** 2

    def sgd_batch(self, X, y):
        # X shape: (batch_size, input_features); y shape: (batch_size, output_features)
        output = X
        for l in self.layers:
            output = l.forward(output)
        error = output - y
        for l in reversed(self.layers):
            error = l.backward(error)
        # Compute mean squared error over the batch.
        return np.mean((output - y) ** 2)

    def full_forward_pass(self, X, Y):
        total_loss = 0
        sample_count = 0
        for i in range(X.shape[0]):
            output = X[i, :]
            for l in self.layers:
                output = l.forward(output)
            total_loss += (Y[i] - output) ** 2
            sample_count += 1
        return total_loss / sample_count

# Data functions remain unchanged.
def generate_xor_data(num_samples=1000, noise_level=0.1):
    X = np.random.rand(num_samples, 2)
    X = 2 * X - 1
    y_clean = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(float).reshape(-1, 1)
    noise = np.random.normal(0, noise_level, size=y_clean.shape)
    y = y_clean + noise
    y = np.clip(y, 0, 1)
    return X, y

def split_data(X, y, test_size=0.2):
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test

def visualize_data(X, y):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='coolwarm', 
                         alpha=0.7, s=50, edgecolors='k')
    plt.colorbar(scatter, label='Target Value')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.title('XOR Problem Data')
    plt.xlabel('Input x₁')
    plt.ylabel('Input x₂')
    plt.show()

np.random.seed(42)
num_samples = 400
X, y = generate_xor_data(num_samples, noise_level=0.05)
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

train_losses = []
test_losses = []
nn = NeuralNetwork(2, 3, 6, 1)
batch_size = 32
num_epochs = 500

for epoch in tqdm(range(num_epochs), desc="Training epochs"):
    indices = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train_shuffled[i:i+batch_size]
        y_batch = y_train_shuffled[i:i+batch_size]
        loss = nn.sgd_batch(X_batch, y_batch)
    # Optionally record loss using full forward pass.
    train_losses.append(nn.full_forward_pass(X_train, y_train))
    test_losses.append(nn.full_forward_pass(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(train_losses, 'b-', linewidth=2, label='Training Loss')
plt.plot(test_losses, 'r-', linewidth=2, label='Test Loss')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Loss Over Epochs', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(bottom=0)
plt.legend(fontsize=10)
plt.annotate(f'Final test loss: {float(train_losses[-1]):.4f}', 
             xy=(len(train_losses)-1, train_losses[-1]),
             xytext=(len(train_losses)-2, train_losses[-1]*1.2),
             arrowprops=dict(arrowstyle='->'))
plt.tight_layout()
plt.show()
