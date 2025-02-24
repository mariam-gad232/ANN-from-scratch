import random
import math
import numpy as np

def tanh(x):
    return np.tanh(x)
def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

i1, i2 = 0.05, 0.10   #input
target_o1, target_o2 = 0.01, 0.99    #target 
w1, w2, w3, w4, w5, w6, w7, w8 = np.random.uniform(-0.5, 0.5, 8)  #weights
b1, b2 = 0.5, 0.7   #bias
learning_rate = 0.1

for epoch in range(10000):
    # forward pass
    h1_input = (i1 * w1) + (i2 * w3) + b1
    h2_input = (i1 * w2) + (i2 * w4) + b1

    h1_output = tanh(h1_input)
    h2_output = tanh(h2_input)

    o1_input = (h1_output * w5) + (h2_output * w7) + b2
    o2_input = (h1_output * w6) + (h2_output * w8) + b2

    o1_output = tanh(o1_input)
    o2_output = tanh(o2_input)

    #calculate error
    error_o1 = target_o1 - o1_output
    error_o2 = target_o2 - o2_output
    total_error = abs(error_o1) + abs(error_o2)

    if total_error < 0.001:
        break

    # backpropagation
    delta_o1 = error_o1 * tanh_derivative(o1_input)
    delta_o2 = error_o2 * tanh_derivative(o2_input)

    delta_h1 = (delta_o1 * w5 + delta_o2 * w6) * tanh_derivative(h1_input)
    delta_h2 = (delta_o1 * w7 + delta_o2 * w8) * tanh_derivative(h2_input)

    # update weights and biases
    w5 += learning_rate * delta_o1 * h1_output
    w6 += learning_rate * delta_o2 * h1_output
    w7 += learning_rate * delta_o1 * h2_output
    w8 += learning_rate * delta_o2 * h2_output

    w1 += learning_rate * delta_h1 * i1
    w2 += learning_rate * delta_h2 * i1
    w3 += learning_rate * delta_h1 * i2
    w4 += learning_rate * delta_h2 * i2

    b1 += learning_rate * (delta_h1 + delta_h2)
    b2 += learning_rate * (delta_o1 + delta_o2)

# output 
print("Final output:")
print("o1:", o1_output)
print("o2:", o2_output)
