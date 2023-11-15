import numpy as np


def sigmoid(beta, x):
    return 1 / (1 + np.exp(-2 * beta * x))


def sigmoid_derivative(beta, x):
    return (2 * beta * np.exp(-2 * beta * x))/(1 + np.exp(-2 * beta * x)) ** 2

