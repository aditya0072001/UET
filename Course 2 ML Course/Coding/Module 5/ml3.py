# Activation functions in python

# Importing libraries
import numpy as np

# Sigmoid function

def sigmoid(x):
    return 1/(1+np.exp(-x))

# tanh function

def tanh(x):
    return np.tanh(x)

# ReLU function

def relu(x):
    return np.maximum(0,x)

# softmax function

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

# Seeing output of activation functions

x = np.array([1,2,3,4,5])
print("Sigmoid: ",sigmoid(x))
print("tanh: ",tanh(x))
print("ReLU: ",relu(x))
print("Softmax: ",softmax(x))

