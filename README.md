# Tanh Neural Network with Backpropagation

This repository contains a simple implementation of a neural network using the hyperbolic tangent (`tanh`) activation function. The network consists of:
- Two input neurons
- Two hidden neurons
- Two output neurons
- Weights initialized randomly within the range [-0.5, 0.5]
- Bias values `b1 = 0.5` and `b2 = 0.7`
- Backpropagation to train the network until the output is close to the target values.

## Features
- Uses `tanh` as the activation function.
- Implements backpropagation with gradient descent.
- Random initialization of weights.
- Stops training when the total error is below `0.001`.

## Installation
Clone this repository:
```
git clone https://github.com/mariam-gad232/ANN-from-scratch.git
```

## Code Overview
-Forward Pass: Computes outputs using weighted sums and the tanh activation function.
-Error Calculation: Computes the difference between predicted and target outputs.
-Backpropagation: Updates weights using the derivative of tanh and the computed error.
-Stopping Condition: Stops training when the error is below 0.001.

## Example Output
```
Final output of the network:
o1: 0.0101
o2: 0.9898
```







