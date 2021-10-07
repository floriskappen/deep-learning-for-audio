# Computation in neural networks

## The components of an artificial neural network (ANN)

- Neurons
- Input, hidden, output layers
- Weighted connections
- Activation function

![Untitled](0.png)

### The multilayer perceptron (MLP)

![Untitled](1.png)

### Computation in MLP

- Weights
- Net inputs (sum of weighted inputs)
- Activations (output of neurons to next layer)

![Untitled](2.png)

Weights - represented in a matrix

![Untitled](3.png)

Net input - with matrix multiplication

![Untitled](4.png)

Activation formula

Computations in MLP

h = layer

W = matrix multiplication

![Untitled](5.png)

1st layer

![Untitled](6.png)

2nd layer

![Untitled](7.png)

3rd layer

### Sample computation

![Untitled](8.png)

Step 1 (1st layer) - The inputs are 0.8 and 1

![Untitled](9.png)

Step 2 (2nd layer) - Net input by matrix multiplication

![Untitled](10.png)

Step 3 (2nd layer) - Pass net input through activation function of choice (eg. Sigmoid in this case)

![Untitled](11.png)

Step 4 (3rd layer) - Matrix multiplication between the activations from the 2nd layer & the weights matrix for the connections between the second and the third layers

![Untitled](12.png)

Step 5 (3rd layer) - Pass the result through the activation function of choice (eg. Sigmoid in this case)

## Takeaway points

- ANNs work for complex problems
- Computation is distributed
- Signal moves from left to right
- Weights, net inputs and activations