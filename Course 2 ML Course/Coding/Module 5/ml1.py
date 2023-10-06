# A simple neural network

# import libraries
import numpy as np

# input data

input_data = np.array([2, 3])

# Neural network parameters

weights = np.array([0.1,0.8])

# Computing output

output = np.dot(input_data, weights)

print(output)

