import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

alpha = 0.1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(a):
    return a * (1-a)
def relu(x):
    return np.maximum(0, x)
# For hidden layers
def relu_derivative(a):
    return np.where(a > 0, 1, 0)
def activate_hidden(x):
    return relu(x)

def derivative_hidden(a):
    return relu_derivative(a)

# For output layer
def activate_output(x):
    return sigmoid(x)

def derivative_output(a):
    return sigmoid_derivative(a)


class Neuron:
    def __init__(self, num_weights, output=False):
        self.weights = np.random.randn(num_weights) * 0.1
        self.bias = np.random.randn() * 0.1
        self.last_activation = None  # Added to store activation for backprop
        self.last_input = None  # Added to store input for backprop
        self.output = output
    def forward(self, x):
        self.last_input = x  # Added to store input
        z = np.dot(self.weights, x) + self.bias
        if self.output:
            self.last_activation = activate_output(z)
        else:
            self.last_activation = activate_hidden(z)
        return self.last_activation
    def backward(self, error_from_next_layer):
        global alpha
        layer_error = 0
        if self.output:
            layer_error = error_from_next_layer * derivative_output(self.last_activation)
        else:
            layer_error = error_from_next_layer * derivative_hidden(self.last_activation)
        
        weights_grad = layer_error * self.last_input
        bias_grad = layer_error
        error_to_return = np.dot(self.weights.T, layer_error)
        self.weights -= alpha * weights_grad
        self.bias -= alpha * bias_grad
        return error_to_return

class Layer:
    def __init__(self, previous_layer_neurons, num_neurons, output=False):
        self.previous_layer_neurons = previous_layer_neurons
        self.num_neurons = num_neurons
        self.neurons = [Neuron(self.previous_layer_neurons,output=output) for i in range(num_neurons)]
        self.last_input = None  # Added to store layer input
        self.last_output = None  # Added to store layer output
    def forward(self, x):
        self.last_input = x  # Added to store input
        outputs = [n.forward(x) for n in self.neurons]
        self.last_output = np.array(outputs)  # Convert to numpy array
        return self.last_output
    def backward(self, error_from_next_layer):  # Changed parameter name and removed x       
       errors_to_return = np.zeros(self.previous_layer_neurons)
       for i, neuron in enumerate(self.neurons):
           neuron_error = neuron.backward(error_from_next_layer[i])  # Removed x parameter
           errors_to_return += neuron_error
       
       return errors_to_return
class NeuralNetwork():
    def __init__(self, input_features, num_layers, neurons_per_layer, output_features):
        self.input_features = input_features
        self.num_layers = num_layers
        self.output_features = output_features
        self.layers = []
        self.layers.append(Layer(input_features, neurons_per_layer)) # Input layer
        for _ in range(num_layers - 2):
            self.layers.append(Layer(neurons_per_layer, neurons_per_layer))
        self.layers.append(Layer(neurons_per_layer, output_features, output=True)) # Output layer
    def sgd(self, x, y):
        # forward pass
        output = x
        for l in self.layers:
            output = l.forward(output)
        # backward pass
        error = output - y
        for l in reversed(self.layers):
            error = l.backward(error)
        return (output - y) ** 2
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

def generate_xor_data(num_samples=1000, noise_level=0.1):
    """
    Generate data for the XOR problem with optional noise.
    XOR (exclusive OR) is a classic non-linear problem that simple linear models cannot solve.
    
    Args:
        num_samples: Number of data points to generate
        noise_level: Standard deviation of Gaussian noise to add
        
    Returns:
        X: Input features, shape (num_samples, 2)
        y: Target outputs, shape (num_samples, 1)
    """
    # Generate random inputs between 0 and 1
    X = np.random.rand(num_samples, 2)
    
    # Scale inputs to be between -1 and 1
    X = 2 * X - 1
    
    # Generate clean XOR outputs (1 if x1 and x2 have different signs, 0 otherwise)
    y_clean = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(float).reshape(-1, 1)
    
    # Add noise to outputs
    noise = np.random.normal(0, noise_level, size=y_clean.shape)
    y = y_clean + noise
    
    # Clip outputs to be between 0 and 1 (since we're using sigmoid activation)
    y = np.clip(y, 0, 1)
    
    return X, y

def split_data(X, y, test_size=0.2):
    """
    Split data into training and test sets.
    
    Args:
        X: Input features
        y: Target outputs
        test_size: Fraction of data to use for testing
        
    Returns:
        X_train, X_test, y_train, y_test: Split datasets
    """
    # Determine split index
    split_idx = int(len(X) * (1 - test_size))
    
    # Split the data
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test

def visualize_data(X, y):
    """
    Visualize the XOR data.
    
    Args:
        X: Input features
        y: Target outputs
    """
    plt.figure(figsize=(10, 8))
    
    # Create a scatter plot with points colored by their target value
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

# Generate the data
np.random.seed(42)  # For reproducibility
num_samples = 500
X, y = generate_xor_data(num_samples, noise_level=0.05)

# Split into training and test sets
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

train_losses = []
test_losses = []
nn = NeuralNetwork(2, 3, 3, 1)
# Training loop with progress bar
batch_size = 32
num_epochs = 50
total_iterations = num_epochs * (len(X_train) // batch_size)

for epoch in tqdm(range(num_epochs), desc="Training epochs"):
    # Shuffle training data at the start of each epoch
    indices = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]
    
    # Process mini-batches
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train_shuffled[i:i+batch_size]
        y_batch = y_train_shuffled[i:i+batch_size]
        
        # Process each sample in the mini-batch
        batch_loss = 0
        for j in range(len(X_batch)):
            batch_loss += nn.sgd(X_batch[j], y_batch[j])
        
        # Record losses every few batches to avoid slowing down training
        if i % (batch_size * 5) == 0:
            train_losses.append(nn.full_forward_pass(X_train, y_train))
            test_losses.append(nn.full_forward_pass(X_test, y_test))



plt.figure(figsize=(10, 6))
plt.plot(train_losses, 'b-', linewidth=2, label='Training Loss')
plt.plot(test_losses, 'r-', linewidth=2, label='Test Loss')
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('SGD Loss Over Iterations', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(bottom=0)
plt.legend(fontsize=10)
plt.annotate(f'Final test loss: {float(train_losses[-1]):.4f}', 
             xy=(len(train_losses)-1, train_losses[-1]),
             xytext=(len(train_losses)-2, train_losses[-1]*1.2),
             arrowprops=dict(arrowstyle='->'))
plt.tight_layout()
plt.show()