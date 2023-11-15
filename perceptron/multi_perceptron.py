import copy
import math
import sys
import time
from functools import partial
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

import numpy as np
import random


# MAX_XB nos permite decidir si la funcion math.exp(-2 * x * beta) da overflow.
# Resulta de resolver la inecuacion: math.exp(-2 * x * beta)  < MAX_FLOAT
# No solo nos permite evitar el overflow, sino que tambien es mas eficiente dado
# que en muchos casos evita hacer math.exp(...). Ej: para limit=100 pasa de 42 segundos
# a 36 segundos.
MAX_XB = math.floor(math.log(sys.float_info.max) / -2) + 2

# MAX_X_RANGE permite evitar hacer el cálculo de math.exp(-2 * x * beta), en los
# casos que sabemos que la respuesta va a dar o muy cercano a 1 o muy cercano a 0.
# Resulta de resolver la ecuacion: 1 / (1 + math.exp(-2 * x * beta)) = 0.999
# Tambien se usa que la funcion es impar.
# Ej: para limit=100 pasa de 33.88 segundos a 28.74 segundos

MAX_X_RANGE = math.log(1/0.999 - 1)


def theta_logistic(beta, x):
    # Evitamos el overflow
    if x < 0 and x * beta < MAX_XB:
        return 0        # 1/inf = 0

    # TODO: check si es seguro hacer esto
    # Eficiencia: evitamos hacer el calculo si ya sabemos que tiende a 0 o 1
    if x > MAX_X_RANGE / (-2 * beta):
        return 0.999
    elif x < MAX_X_RANGE / (2 * beta):
        return 0.001

    return 1 / (1 + math.exp(-2 * x * beta))


def theta_logistic_derivative(beta, x):
    theta_result = theta_logistic(beta, x)
    return 2 * beta * theta_result * (1 - theta_result)

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


class NeuronLayer:
    def __init__(self,
                 previous_layer_neuron_amount,
                 current_layer_neurons_amount,
                 activation_function,
                 activation_function_derivative,
                 lower_weight,
                 upper_weight,
                 alpha = 0.001,
                 beta1 = 0.9,
                 beta2 = 0.999,
                 epsilon = 0.000000001
         ):
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

        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.t = 0
        self.m_t = 0
        self.v_t = 0

    def compute_activation(self, prev_input):

        # guardamos el dot producto dado que lo vamos a usar aca y en el backpropagation
        self.excitement = np.dot(self.weights, prev_input)

        self.output = self.activation_function(self.excitement)
        self.output_derivative = self.activation_function_derivative(self.excitement)

        return self.output  # Se ejecuta la funcion sobre cada elemento del arreglo

    def update_weights(self, delta_w, alpha):
        new_delta = delta_w + alpha * self.prev_delta
        self.weights += new_delta
        self.prev_delta = new_delta

    def update_weights_adam(self, delta_w, error):
        self.t += 1

        gt = self.output_derivative * error

        self.m_t = self.beta1 * self.m_t + (1 - self.beta1) * gt
        self.v_t = self.beta2 * self.v_t + (1 - self.beta2) * np.power(gt, 2)

        final_m_t = self.m_t / (1 - self.beta1 ** self.t)
        final_v_t = self.v_t / (1 - self.beta2 ** self.t)

        self.weights = self.weights - self.alpha * final_m_t / (np.sqrt(final_v_t) + self.epsilon)



class MultiPerceptron:

    def __init__(self, layer_configuration, activation_function, derivative_activation_function, learning_constant, beta):

        self.activation_function = np.vectorize(partial(activation_function, beta))
        self.derivative_activation_function = np.vectorize(partial(derivative_activation_function, beta))

        self.learning_constant = learning_constant
        self.input = None

        # Variables usadas en compute_error_parallel
        self.error_calc_items = None

        # Caclculamos el rango de valores iniciales para los weights
        upper_weight = math.log(1 / 0.98 - 1) / (-2 * beta)
        lower_weight = - upper_weight

        self.layers: [NeuronLayer] = []
        for i in range(len(layer_configuration)):
            prev = max(0, i-1)      # Caso: primera capa que no podes tener prev = -1
            self.layers.append(NeuronLayer(layer_configuration[prev], layer_configuration[i], self.activation_function, lower_weight, upper_weight))

    def forward_propagation(self, input_data):
        current = input_data
        self.input = input_data
        for layer in self.layers:
            current = layer.compute_activation(current)

        return current

    def update_all_weights(self, delta_w, alpha):  # [matriz1,matriz2,matriz3]
        for idx, layer in enumerate(self.layers):
            layer.update_weights(delta_w[idx], alpha)

    def compute_error(self, data_input, expected_outputs):

        error_vector = []

        for i, o in zip(data_input, expected_outputs):
            output_result = self.forward_propagation(i)
            error_vector.append(np.power(o - output_result, 2))

        total = 0
        for elem in error_vector:
            total += sum(elem)

        return 0.5 * total

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

    def train(self, limit, alpha, input_data, expected_output):
        i = 0
        min_error = float("inf")
        while i < limit:
            # usamos todos los datos
            for i, o in zip(input_data, expected_output):

                result = self.forward_propagation(i)
                delta_w = self.back_propagation(o, result)

                # Actualizamos los pesos
                self.update_all_weights(delta_w, alpha)

                error = self.compute_error(input_data, expected_output)

                if error < min_error:
                    min_error = error

            i += 1

        return min_error

    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights.append(copy.deepcopy(layer.weights))
        return weights

    def test(self, input_test_data, expected_output, epsilon=0.05):
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        for input_data, outputs in zip(input_test_data, expected_output):
            results = self.forward_propagation(input_data)
            for result, expected_output in zip(results, outputs):
                if expected_output == 1:
                    if math.fabs(expected_output - result) < epsilon:
                        true_positive += 1
                    else:
                        false_negative += 1
                else:
                    if math.fabs(expected_output - result) < epsilon:
                        true_negative += 1
                    else:
                        false_positive += 1

        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive+false_negative)
        f1_score = None
        if precision + recall != 0:
            f1_score = (2 * precision * recall) / (precision + recall)

        return accuracy, precision, recall, f1_score

    @staticmethod
    def initialize_metrics(metrics):
        metrics["error"] = []
        metrics["iteration"] = 0
