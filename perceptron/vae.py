from functools import partial

import numpy as np

from perceptron.functions import sigmoid, sigmoid_derivative
from perceptron.multi_perceptron import NeuronLayer
from perceptron.optimizers import ADAM, Optimizer
from training_data.font import fonts


def reconstruction_loss(generated_output, expected_output):
    return sum(np.power(generated_output - expected_output, 2))


def kl(mean, std):
    return -0.5 * np.sum(1 + std - mean ** 2 - np.exp(std))


def total_loss(mean, std, generated_output, expected_outputs):
    return reconstruction_loss(generated_output, expected_outputs) + kl(mean,std)


def count_error(output, expected_output):
    incorrect_pixels = 0
    for i in range(len(output)):
        val = output[i]

        if round(val) != expected_output[i]:
            incorrect_pixels += 1
    return incorrect_pixels


class SamplingLayer:
    def compute_activation(self, prev_input):
        self.mean, self.std = np.array_split(prev_input, 2)

        self.epsilon = np.random.standard_normal(int(prev_input.shape[0] / 2)).reshape(-1, 1)

        # Z         = mu        + sigma * epsilon
        self.output = self.mean + np.dot(self.std, self.epsilon)

        self.excitement = np.concatenate((self.mean, self.std), axis=0)

        return self.output


class VAE:
    def __init__(self,
                 layer_configuration,
                 activation_function,
                 derivative_activation_function,
                 optimizer,
                 optimizer_args
                 ):

        self.activation_function = activation_function
        self.derivative_activation_function = derivative_activation_function

        # Caclculamos el rango de valores iniciales para los weights
        upper_weight = 1
        lower_weight = - upper_weight

        self.layers: [NeuronLayer] = []

        # = = = Generamos encoder = = =
        for i in range(1, len(layer_configuration) - 1):
            prev = i - 1  # Caso: primera capa que no podes tener prev = -1

            # Generamos nueva capa con las dimensiones apropiadas
            self.layers.append(NeuronLayer(
                layer_configuration[prev],
                layer_configuration[i],
                self.activation_function,
                lower_weight,
                upper_weight,
                optimizer(*optimizer_args)
            ))

        #  = = = Generamos espacio latente = = =
        self.latent_idx = len(layer_configuration) - 1

        self.layers.append(
            # Mean and Standard deviation
            # Los generamos como 1 capa para facilitar las cuentas, luego lo dividiremos
            NeuronLayer(
                layer_configuration[-2],
                layer_configuration[-1] * 2,
                self.activation_function,
                lower_weight,
                upper_weight,
                optimizer(*optimizer_args)
            ),
        )

        # Generamos el Sampling Layer
        self.layers.append(
            SamplingLayer()
        )

        #  = = = Generamos decoder = = =
        for i in reversed(range(len(layer_configuration) - 1)):
            following = i + 1  # Caso: primera capa que no podes tener prev = -1

            # Generamos nueva capa con las dimensiones apropiadas
            self.layers.append(NeuronLayer(
                layer_configuration[following],
                layer_configuration[i],
                self.activation_function,
                lower_weight,
                upper_weight,
                optimizer(*optimizer_args)
            ))

    def forward_propagation(self, input_data):
        current = input_data
        self.input = input_data

        for idx, layer in enumerate(self.layers):
            current = layer.compute_activation(current)

        return current
    def backward_propagation(self, expected_output, generated_output):
        delta_w = []

        # = = = = Calculamos el delta W de la primera capa del decoder = = = =

        prev_delta = (expected_output - generated_output) * self.derivative_activation_function(self.layers[-1].excitement)
        delta_w.append(prev_delta.reshape(-1, 1) @ np.transpose(self.layers[-2].output.reshape(-1, 1)))

        # = = = = Calculamos el delta W de las capas ocultas del decoder = = = =
        for idx in range(len(self.layers) - 2, self.latent_idx, -1):
            delta = np.dot(prev_delta, self.layers[idx + 1].weights) * self.derivative_activation_function(self.layers[idx].excitement)
            delta_w.append(delta.reshape(-1, 1) @ np.transpose(self.layers[idx - 1].output.reshape(-1, 1)))
            prev_delta = delta

        # = = = = Calculamos el delta W de la primera capa del encoder = = = =

        # Reparametrization trick (que ni entiendo que pingo hace)
        mean = np.dot(prev_delta, self.layers[self.latent_idx + 1].weights)
        std = self.layers[self.latent_idx].epsilon.reshape(-1,) * mean
        prev_delta = np.concatenate((mean, std), axis=0)

        delta_w.append(prev_delta.reshape(-1, 1) @ np.transpose(self.layers[self.latent_idx - 2].output.reshape(-1, 1)))

        # = = = = Calculamos el delta W de las capas ocultas del encoder = = = =
        for idx in range(self.latent_idx - 2, 0, -1):
            delta =  np.dot(prev_delta, self.layers[idx + 1].weights) * self.derivative_activation_function(self.layers[idx].excitement)
            delta_w.append(delta.reshape(-1, 1) @ np.transpose(self.layers[idx - 1].output.reshape(-1, 1)))
            prev_delta = delta

        #  = = = = Calculamos el delta W de la capa inicial = = = =
        delta = np.dot(prev_delta, self.layers[1].weights) * self.derivative_activation_function(self.layers[0].excitement)
        delta_w.append(delta.reshape(-1, 1) @ np.transpose(self.input.reshape(-1, 1)))

        delta_w.reverse()

        return delta_w

    def update_all_weights(self, delta_w):
        for idx, layer in enumerate(self.layers):
            if idx != self.latent_idx:
                layer.update_weights(delta_w.pop(0))

    def train(self, limit, train_data):
        i = 0
        min_error = float("inf")
        while i < limit:
            error = 0

            for elem in train_data:
                result = self.forward_propagation(elem)
                delta_w = self.backward_propagation(elem, result)
                self.update_all_weights(delta_w)

                error += total_loss(self.layers[self.latent_idx].mean, self.layers[self.latent_idx].std, result, elem)

            if error < min_error:
                min_error = error

            if i % 200 == 0:
                print(f"Error {i}: {min_error}")

            i += 1
        return min_error

    def test(self, input_data, expected_output):
        total_incorrect_pixels = 0
        total_incorrect = 0

        for a, b in zip(input_data, expected_output):
            result = self.forward_propagation(a)

            incorrect = count_error(result, b)
            total_incorrect_pixels += incorrect

            if incorrect > 1:
                total_incorrect += 1
                print(f"\n> Not passed! {incorrect} pixels incorrect!")
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
                print(f"> Passed! {incorrect} pixels incorrect!")

        print("\n- = - = - TESTS FINISHED - = - = -\n")
        print(f"Total incorrect pixels: {total_incorrect_pixels}")
        print(f"Success rate: { (len(input_data) - total_incorrect) / len(input_data)}")

        return total_incorrect



vae = VAE([35, 30, 25, 20, 15, 10, 5, 2], partial(sigmoid, 1), partial(sigmoid_derivative, 1), ADAM, [0.001, 0.9, 0.999, 1e-8])

vae.train(10000, fonts)
vae.test(fonts, fonts)



