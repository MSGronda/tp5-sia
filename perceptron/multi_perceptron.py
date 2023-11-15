import copy
import math
import sys
import time
from functools import partial
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

import numpy as np
import random


def sigmoid(beta, x):
    return 1 / (1 + np.exp(-2 * beta * x))


def sigmoid_derivative(beta, x):
    return (2 * beta * np.exp(-2 * beta * x))/(1 + np.exp(-2 * beta * x)) ** 2


def convert_data(data_input, data_output):
    new_input = []
    new_output = []

    for i, o in zip(data_input, data_output):
        new_input.append(np.array(i))
        new_output.append(np.array(o))

    return np.array(new_input), np.array(new_output)


def compute_error_single(data):

    # Esta funcion combina los metodos compute_error de MultiPerceptron y
    # compute_activation de NueronLayer. Esta disenado para ser usado
    # de forma paralela (como en compute_error_parallel).

    weights, activation_function, data_input, expected_output = data[0], data[1], data[2], data[3]

    current = data_input
    for weight in weights:
        current = activation_function(np.dot(weight, current))

    return np.power(expected_output - current, 2)


def check_valid(output, expected_output):
    incorrect_pixels = 0
    for i in range(len(output)):
        val = output[i]

        if round(val) != expected_output[i]:
            incorrect_pixels += 1

    return True if incorrect_pixels <= 1 else False


class NeuronLayer:
    def __init__(self,
                 previous_layer_neuron_amount,
                 current_layer_neurons_amount,
                 activation_function,
                 activation_function_derivative,
                 lower_weight,
                 upper_weight,
                 alpha=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8):

        self.excitement = None
        self.output = None
        self.output_derivative = None
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

        # Se genera una matrix de (current_layer_neurons_amount x previous_layer_neuron_amount)
        weights = []
        for i in range(current_layer_neurons_amount):
            weights.append([])
            for j in range(previous_layer_neuron_amount):
                weights[i].append(random.uniform(lower_weight, upper_weight))
        self.weights = np.array(weights)

        self.prev_delta = 0

        # Variables para optimizacion ADAM
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m_t = np.zeros([current_layer_neurons_amount, previous_layer_neuron_amount])
        self.v_t = np.zeros([current_layer_neurons_amount, previous_layer_neuron_amount])
        self.t = 0

    def compute_activation(self, prev_input):

        # guardamos el dot producto dado que lo vamos a usar aca y en el backpropagation
        self.excitement = np.dot(self.weights, prev_input)

        self.output = self.activation_function(self.excitement)
        self.output_derivative = self.activation_function_derivative(self.excitement)

        return self.output  # Se ejecuta la funcion sobre cada elemento del arreglo

    def update_weights(self, delta_w):
        new_delta = delta_w + self.alpha * self.prev_delta
        self.weights += new_delta
        self.prev_delta = new_delta

    def update_weights_adam(self, delta_w):
        self.t += 1

        self.m_t = self.beta1 * self.m_t + (1 - self.beta1) * delta_w
        self.v_t = self.beta2 * self.v_t + (1 - self.beta2) * np.power(delta_w, 2)

        final_m_t = self.m_t / (1 - self.beta1 ** self.t)
        final_v_t = self.v_t / (1 - self.beta2 ** self.t)

        self.weights += self.alpha * final_m_t / (np.sqrt(final_v_t) + self.epsilon)


class MultiPerceptron:

    def __init__(self, layer_configuration, activation_function, derivative_activation_function, learning_constant, beta):

        self.activation_function = partial(activation_function, beta)
        self.derivative_activation_function = partial(derivative_activation_function, beta)

        self.learning_constant = learning_constant
        self.input = None

        # Variables usadas en compute_error_parallel
        self.error_calc_items = None

        # Caclculamos el rango de valores iniciales para los weights
        upper_weight = 1
        lower_weight = - upper_weight

        self.layers: [NeuronLayer] = []
        for i in range(len(layer_configuration)):
            prev = max(0, i-1)      # Caso: primera capa que no podes tener prev = -1
            self.layers.append(NeuronLayer(layer_configuration[prev], layer_configuration[i], self.activation_function, self.derivative_activation_function,  lower_weight, upper_weight))

    def forward_propagation(self, input_data):
        current = input_data
        self.input = input_data
        for layer in self.layers:
            current = layer.compute_activation(current)

        return current

    def update_all_weights(self, delta_w):
        for idx, layer in enumerate(self.layers):
            layer.update_weights_adam(delta_w[idx])

    def compute_error(self, data_input, expected_outputs):
        error = 0
        for i, o in zip(data_input, expected_outputs):
            output_result = self.forward_propagation(i)
            error += sum(np.power(o - output_result, 2))

        return 0.5 * error

    def compute_error_parallel(self, data_input, expected_outputs):

        # Este metodo permite calcular el error de forma paralela.
        # MG: de lo que tengo entendido, es la unica parte del metodo train
        # que se puede paralelizar.
        # Se uso ThreadPool que usa threads en vez de Pool que usa procesos porque
        # es demasiado caro generar nuevos procesos y termina siendo mucho peor.
        # Performance: pasa de 2.8s para procesar 10000 elementos a 2.1s.

        weights = self.get_weights()

        if self.error_calc_items is None:
            self.error_calc_items = []
            for i, o in zip(data_input, expected_outputs):
                # Usamos todas referencias asi no hay que re generar el arreglo de items.
                self.error_calc_items.append([weights, self.activation_function, i, o])
        else:
            # Updateamos las referencias a los nuevos pesos
            # No encontre mejor manera para hacer esto :(
            for i in range(len(self.error_calc_items)):
                self.error_calc_items[i][0] = weights

        total = 0
        with ThreadPool() as pool:
            results = pool.imap_unordered(compute_error_single, self.error_calc_items)

            for elem in results:
                total += sum(elem)

        return 0.5 * total

    def back_propagation(self, expected_output, generated_output) -> list:
        delta_w = []

        # Calculamos el delta W de la capa de salida
        prev_delta = (expected_output - generated_output) * self.derivative_activation_function(self.layers[-1].excitement)
        delta_w.append(self.learning_constant * prev_delta.reshape(-1, 1) @ np.transpose(self.layers[-2].output.reshape(-1, 1)))

        # Calculamos el delta W de las capas ocultas
        for idx in range(len(self.layers) - 2, 0, -1):
            delta = np.dot(prev_delta, self.layers[idx + 1].weights) * self.derivative_activation_function(self.layers[idx].excitement)
            delta_w.append(self.learning_constant * delta.reshape(-1, 1) @ np.transpose(self.layers[idx - 1].output.reshape(-1, 1)))
            prev_delta = delta

        # Calculamos el delta W de la capa inicial
        delta = np.dot(prev_delta, self.layers[1].weights) * self.derivative_activation_function(self.layers[0].excitement)
        delta_w.append(self.learning_constant * delta.reshape(-1, 1) @ np.transpose(self.input.reshape(-1, 1)))

        delta_w.reverse()

        return delta_w

    def train(self, limit, input_data, expected_output):
        i = 0
        min_error = float("inf")
        while i < limit:
            # usamos todos los datos
            for a, b in zip(input_data, expected_output):
                result = self.forward_propagation(a)

                delta_w = self.back_propagation(b, result)

                error = self.compute_error(input_data, expected_output)

                self.update_all_weights(delta_w)

                if error < min_error:
                    min_error = error

            if i % 50 == 0:
                print(f"Error {i}: {min_error}")

            i += 1
        return min_error

    def test(self, input_data, expected_output):
        for a, b in zip(input_data, expected_output):
            result = self.forward_propagation(a)

            if not check_valid(result, b):
                print("Not passed!")
            else:
                print("Passed!")

    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights.append(copy.deepcopy(layer.weights))
        return weights
