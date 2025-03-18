import numpy as np
import matplotlib.pyplot as plt

w1 = 0.5
w2 = 0.7
b = 0.2
alpha = 0.2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def loss(answer, expected):
    return (answer - expected) ** 2

def epoch(X, y):
    global w1, w2, b, alpha
    total_loss = 0
    sample_count = 0
    w1_grad = 0
    w2_grad = 0
    b_grad = 0
    for i in range(X.shape[0]):
        sample = X[i, :]
        z = w1 * sample[0] + w2 * sample[1] + b
        a = sigmoid(z)
        total_loss += loss(a, y[i])
        w1_grad += -2 * (y[i] - a) * a * (1 - a) * sample[0]
        w2_grad += -2 * (y[i] - a) * a * (1 - a) * sample[1]
        b_grad += -2 * (y[i] - a) * a * (1 - a)
        sample_count += 1
    w1_grad /= sample_count
    w2_grad /= sample_count
    b_grad /= sample_count
    w1 -= alpha * w1_grad
    w2 -= alpha * w2_grad
    b -= alpha * b_grad
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

for i in range(1000):
    e = epoch(X_train, y_train)
    if i % 100 == 0:
        print(e)
# print(f"Training set: {X_train.shape[0]} samples")
# print(f"Test set: {X_test.shape[0]} samples")

# # Visualize the data
# visualize_data(X, y)

# # Example of how to access a single training sample
# print("\nExample training sample:")
# sample_idx = 0
# print(f"Input: [{X_train[sample_idx, 0]:.4f}, {X_train[sample_idx, 1]:.4f}]")
# print(f"Target: {y_train[sample_idx, 0]:.4f}")