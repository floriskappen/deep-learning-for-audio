# Training a neural network: Backward propagation and gradient descent

## Training a NN

- Tweak weights of the connections
- Feed training data (input + target) to the network
- Iterative adjustments

### Forward propagation

![Untitled](0.png)

### Back propagation

![Untitled](1.png)

1. Make prediction

2. Calculate error

We need an error (loss) function for that.

![Untitled](2.png)

3. Calculate gradient of the error function

We think of a NN as a very complex function which is dependant on 2 variables. The x, which is the input, and the W, which are the weights. 

![Untitled](3.png)

The error function is a function of p and y. p, being the prediction, is a function that's the result of F(x, W). 

## Gradient descent

- Take a step in the opposite direction to gradient
- Step = Learning rate

![Untitled](4.png)