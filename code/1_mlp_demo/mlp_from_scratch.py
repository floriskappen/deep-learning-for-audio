import numpy as np
from random import random

# Save activations and derivatives
# Implement backpropagation
# Implement gradient descent
# Implement train
# Train our network with some dummy dataset
# Make some predictions

class MLP:
  def __init__(self, num_inputs=3, num_hidden=[3, 5], num_outputs=2):

    self.num_inputs = num_inputs
    self.num_hidden = num_hidden
    self.num_outputs = num_outputs

    layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

    # Initiate random weights
    weights = []
    for i in range(len(layers)-1):
      w = np.random.rand(layers[i], layers[i+1])
      weights.append(w)
    self.weights = weights

    activations = []
    for i in range(len(layers)):
      a = np.zeros(layers[i])
      activations.append(a)
    self.activations = activations

    derivatives = []
    for i in range(len(layers)-1):
      d = np.zeros((layers[i], layers[i+1]))
      derivatives.append(d)
    self.derivatives = derivatives


  def forward_propagate(self, inputs):
    """Computes forward propagation of the network based on input signals.

    Args:
      inputs (ndarray): Input signals
    Returns:
      activations (ndarray): Output values
    """

    activations = inputs
    self.activations[0] = inputs

    for i, w in enumerate(self.weights):
      # Calculate the net inputs
      net_inputs = np.dot(activations, w)

      # Calculate the activations
      activations = self._sigmoid(net_inputs)
      self.activations[i+1] = activations

    return activations


  def back_propagate(self, error, verbose=False):

    for i in reversed(range(len(self.derivatives))):
      activations = self.activations[i+1]
      delta = error * self._sigmoid_derivative(activations) # ndarray([0.2, 0.3]) --> ndarray([[0.2, 0.3]])
      delta_reshaped = delta.reshape(delta.shape[0], -1).T
      current_activations = self.activations[i] # ndarray([0.2, 0.3]) --> ndarray([[0.2], [0.3]])
      current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
      self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
      error = np.dot(delta, self.weights[i].T)
      
      if verbose:
        print(f"Derivatives for W{i}: {self.derivatives[i]}")

    return error


  def gradient_descent(self, learning_rate):

    for i in range(len(self.weights)):
      weights = self.weights[i]
      derivatives = self.derivatives[i]
      weights += + derivatives * learning_rate


  def train(self, inputs, targets, epochs, learning_rate):

    for i in range(epochs):
      sum_error = 0
      for input, target in zip(inputs, targets):

        # Perform forward propagation
        output = self.forward_propagate(input)

        # Calculate error
        error = target - output

        # Back propagation
        self.back_propagate(error)

        # Apply gradient descent
        self.gradient_descent(learning_rate)

        sum_error += self._mse(target, output)

      # Report error
      print(f"Error: {sum_error / len(inputs)} at epoch {i}")


  def _mse(self, target, output):
    return np.average((target - output)**2)


  def _sigmoid_derivative(self, x):
    return x * (1.0 - x)


  def _sigmoid(self, x):
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
  
  # Create a dataset to train a network for the sum operation
  inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)]) # Array([[0.1, 0.2], [0.3, 0.4]])
  targets = np.array([[i[0] + i[1]] for i in inputs]) # Array([[0.3], [0.7]])

  # Create an MLP
  mlp = MLP(2, [5], 1)

  # Train our MLP
  mlp.train(inputs, targets, 100, learning_rate=1)

  # Create dummy data
  input = np.array([0.3, 0.1])

  output = mlp.forward_propagate(input)
  print()
  print()
  print(f"Our network believes that {input[0]} + {input[1]} is equal to {output[0]}")