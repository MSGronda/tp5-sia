import numpy as np
from random import randint, uniform, random


def sigmoid(beta, x):
    return 1 / (1 + np.exp(-2 * beta * x))


def sigmoid_derivative(beta, x):
    return (2 * beta * np.exp(-2 * beta * x))/(1 + np.exp(-2 * beta * x)) ** 2



#flipea n bits del vector de forma aleatoria
def bit_fliping_with_n(vectorized_data,bits_to_flip_number : int):

    bits_seen = set()
    i = 0
    while i < bits_to_flip_number:
        current_bit_idx = randint(0,len(vectorized_data) - 1)
        if current_bit_idx not in bits_seen:
            bits_seen.add(current_bit_idx)
            vectorized_data[current_bit_idx] = 0 if vectorized_data[current_bit_idx] == 1 else 1
            i += 1


#intenta flipear cada bit del conjunto con una probabilidad dada
def bit_fliping_with_probability(vectorized_data,probability : float):
    for i in range(len(vectorized_data)):
        if random() < probability:
            vectorized_data[i] = 0 if vectorized_data[i] == 1 else 1
