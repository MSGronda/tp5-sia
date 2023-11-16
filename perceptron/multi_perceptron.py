import copy
from functools import partial
from multiprocessing.pool import ThreadPool
from typing import Type

import numpy as np
import random

from perceptron.optimizers import Optimizer


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


def add_delta_w(delta_w, inc_delta_w):
    if delta_w is None:
        delta_w = inc_delta_w
    else:
        for i in range(len(delta_w)):
            delta_w[i] += inc_delta_w[i]

    return delta_w


class NeuronLayer:
    def __init__(self,
                 previous_layer_neuron_amount,
                 current_layer_neurons_amount,
                 activation_function,
                 lower_weight,
                 upper_weight,
                 optimizer: Optimizer
                 ):

        self.excitement = None
        self.output = None
        self.activation_function = activation_function

        self.optimizer = optimizer

        # Se genera una matrix de (current_layer_neurons_amount x previous_layer_neuron_amount)
        weights = []
        for i in range(current_layer_neurons_amount):
            weights.append([])
            for j in range(previous_layer_neuron_amount):
                weights[i].append(random.uniform(lower_weight, upper_weight))
        self.weights = np.array(weights)


    def compute_activation(self, prev_input):

        # guardamos el dot producto dado que lo vamos a usar aca y en el backpropagation
        self.excitement = np.dot(self.weights, prev_input)

        self.output = self.activation_function(self.excitement)

        return self.output  # Se ejecuta la funcion sobre cada elemento del arreglo

    def update_weights(self, delta_w):
        self.weights += self.optimizer.calc_delta_w(delta_w)


class MultiPerceptron:

    def __init__(self,
                 layer_configuration,
                 activation_function,
                 derivative_activation_function,
                 optimizer,
                 optimizer_args
                 ):

        self.activation_function = activation_function
        self.derivative_activation_function = derivative_activation_function

        self.input = None

        # Variables usadas en compute_error_parallel
        self.error_calc_items = None

        # Caclculamos el rango de valores iniciales para los weights
        upper_weight = 1
        lower_weight = - upper_weight

        self.layers: [NeuronLayer] = []
        for i in range(len(layer_configuration)):

            prev = max(0, i-1)      # Caso: primera capa que no podes tener prev = -1

            # Generamos nueva capa con las dimensiones apropiadas
            self.layers.append(NeuronLayer(
                layer_configuration[prev],
                layer_configuration[i],
                self.activation_function,
                lower_weight,
                upper_weight,
                optimizer(*optimizer_args)
            ))

    def forward_propagation(self, input_data):
        current = input_data
        self.input = input_data
        for layer in self.layers:
            current = layer.compute_activation(current)

        return current

    def update_all_weights(self, delta_w):
        for idx, layer in enumerate(self.layers):
            layer.update_weights(delta_w[idx])

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

    def back_propagation(self, expected_output, generated_output):
        delta_w = []

        # Calculamos el delta W de la capa de salida
        prev_delta = (expected_output - generated_output) * self.derivative_activation_function(self.layers[-1].excitement)
        delta_w.append(prev_delta.reshape(-1, 1) @ np.transpose(self.layers[-2].output.reshape(-1, 1)))

        # Calculamos el delta W de las capas ocultas
        for idx in range(len(self.layers) - 2, 0, -1):
            delta = np.dot(prev_delta, self.layers[idx + 1].weights) * self.derivative_activation_function(self.layers[idx].excitement)
            delta_w.append(delta.reshape(-1, 1) @ np.transpose(self.layers[idx - 1].output.reshape(-1, 1)))
            prev_delta = delta

        # Calculamos el delta W de la capa inicial
        delta = np.dot(prev_delta, self.layers[1].weights) * self.derivative_activation_function(self.layers[0].excitement)
        delta_w.append(delta.reshape(-1, 1) @ np.transpose(self.input.reshape(-1, 1)))

        delta_w.reverse()

        return delta_w

    def train(self, limit, input_data, expected_output, batch_size):
        i = 0
        min_error = float("inf")
        while i < limit:
            delta_w = None

            # Usamos todos los datos
            if batch_size == len(input_data):
                for a, b in zip(input_data, expected_output):
                    result = self.forward_propagation(a)
                    inc = self.back_propagation(b, result)

                    delta_w = add_delta_w(delta_w, inc)

            # Usamos un subconjunto
            else:
                for _ in range(batch_size):
                    idx = random.randint(0, len(input_data) - 1)
                    result = self.forward_propagation(input_data[idx])
                    inc = self.back_propagation(expected_output[idx], result)

                    delta_w = add_delta_w(delta_w, inc)

            self.update_all_weights(delta_w)

            error = self.compute_error(input_data, expected_output)

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
                print("Res:\t\tExp:")
                i = 0
                for elem in result:
                    print(f"{round(elem)}", end=" ")
                    i += 1
                    if i % 5 == 0:
                        print(f"\t", end="")
                        for j in range(5):
                            print(f"{round(a[j+i-5])}", end=" ")
                        print("")
                print("")
            else:
                print("Passed!\n")

    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights.append(copy.deepcopy(layer.weights))
        return weights
