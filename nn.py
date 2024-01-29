import numpy as np
import matplotlib.pyplot as plt

# Sigmoid activation function
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

# Derivative of the sigmoid function
def sigmoid_derivative(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))

# Initialize parameters
def initialize_parameters(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims) - 1
    
    for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
    return parameters

# Forward propagation
def forward_propagation(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    
    for l in range(1, L + 1):
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        
        Z = np.dot(W, A) + b
        A = sigmoid(Z)
        
        cache = (np.copy(A), np.copy(W), np.copy(b), np.copy(Z))
        caches.append(cache)
    
    return A, caches

# Compute cross-entropy loss
def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -1/m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    cost = np.squeeze(cost)
    return cost

# Backward propagation
# Modify the backward_activation function
def backward_activation(dA, cache):
    A_prev, W, b, Z = cache
    m = A_prev.shape[1]
    
    dZ = dA * sigmoid_derivative(Z)
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

# Modify the backward_propagation function
def backward_propagation(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L - 1]
    
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = backward_activation(dAL, current_cache)
    
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = backward_activation(grads["dA" + str(l + 1)], current_cache)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    
    return grads

# Update parameters using gradient descent
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    
    for l in range(1, L + 1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]
    
    return parameters

# Example training data
np.random.seed(1)
X_train = np.random.randn(3, 100)  # Example features (3 features, 100 samples)
Y_train = np.random.randint(0, 2, (1, 100))  # Example labels (binary classification)

# Neural network architecture
layer_dims = [3, 4, 1]  # Input layer: 3 units, Hidden layer: 4 units, Output layer: 1 unit

# Training parameters
epochs = 10000
learning_rate = 0.01

# Initialize parameters
parameters = initialize_parameters(layer_dims)

# Training loop
cost_history = []
for i in range(epochs):
    # Forward propagation
    AL, caches = forward_propagation(X_train, parameters)
    
    # Compute cost
    cost = compute_cost(AL, Y_train)
    cost_history.append(cost)
    
    # Backward propagation
    grads = backward_propagation(AL, Y_train, caches)
    
    # Update parameters
    parameters = update_parameters(parameters, grads, learning_rate)
    
    if i % 1000 == 0:
        print(f"Cost after iteration {i}: {cost}")

# Plotting the cost history
plt.plot(cost_history)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost over Epochs')
plt.show()
