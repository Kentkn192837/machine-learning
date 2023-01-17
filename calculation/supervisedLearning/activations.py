import numpy as np

def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def softmax(inputs):
    c = np.max(inputs)
    return np.exp(inputs - c) / np.sum(np.exp(inputs - c))
