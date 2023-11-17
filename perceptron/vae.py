from functools import partial

import numpy as np

from perceptron.functions import sigmoid, sigmoid_derivative
from perceptron.multi_perceptron import NeuronLayer
from perceptron.optimizers import ADAM
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
    def __init__(self):
        self.output = None

    def sampling(self, mean, std):
        epsilon = np.random.randn(*mean.shape)
        self.output = mean + std * epsilon
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
        for i in range(len(layer_configuration) - 1):
            prev = max(0, i - 1)  # Caso: primera capa que no podes tener prev = -1

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

        self.layers.append([
            # Mean
            NeuronLayer(
                layer_configuration[self.latent_idx - 1],
                layer_configuration[self.latent_idx],
                self.activation_function,
                lower_weight,
                upper_weight,
                optimizer(*optimizer_args)
            ),

            # Standard deviation
            NeuronLayer(
                layer_configuration[self.latent_idx - 1],
                layer_configuration[self.latent_idx],
                self.activation_function,
                lower_weight,
                upper_weight,
                optimizer(*optimizer_args)
            ),

            # Sampling layer
            SamplingLayer()

        ])

        #  = = = Generamos decoder = = =
        for i in reversed(range(len(layer_configuration) - 1)):
            following = min(len(layer_configuration), i + 1)  # Caso: primera capa que no podes tener prev = -1

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
            if idx != self.latent_idx:
                current = layer.compute_activation(current)
            else:
                mean = layer[0].compute_activation(current)
                std = layer[1].compute_activation(current)
                current = layer[2].sampling(mean, std)

        return current, mean, std

    def backward_propagation(self, expected_output, generated_output):
        delta_w = []

        prev_delta = (expected_output - generated_output) * self.derivative_activation_function(self.layers[-1].excitement)
        delta_w.append(prev_delta.reshape(-1, 1) @ np.transpose(self.layers[-2].output.reshape(-1, 1)))

        # = = = = Calculamos el delta W de las capas ocultas del decoder = = = =
        for idx in range(len(self.layers) - 2, self.latent_idx, -1):
            delta = np.dot(prev_delta, self.layers[idx + 1].weights) * self.derivative_activation_function(self.layers[idx].excitement)
            if idx - 1 == self.latent_idx:
                # se agrega [2] porque estamos en la capa de sampling layer
                delta_w.append(delta.reshape(-1, 1) @ np.transpose(self.layers[idx - 1][2].output.reshape(-1, 1)))  # TODO: reemplazar magic number
            else:
                delta_w.append(delta.reshape(-1, 1) @ np.transpose(self.layers[idx - 1].output.reshape(-1, 1)))
            prev_delta = delta

        prev_delta = prev_delta.reshape(-1, 1).T

        #  = = = = Calculamos el delta W de la capa de espacio latente = = = =
        # TODO: check
        gradient_mean = np.dot(prev_delta, self.layers[self.latent_idx][0].weights.T)
        gradient_std = np.dot(prev_delta, self.layers[self.latent_idx][1].weights.T)

        # Compute the gradients for the mean and std in the latent layer
        delta_w_mean = gradient_mean.reshape(-1, 1) @ np.transpose(self.layers[self.latent_idx - 1].output.reshape(-1, 1))
        delta_w_std = gradient_std.reshape(-1, 1) @ np.transpose(self.layers[self.latent_idx - 1].output.reshape(-1, 1))

        delta_w.append([delta_w_mean, delta_w_std])

        #  = = = = Calculamos el delta W de las capas ocultas del encoder = = = =
        # TODO: Check lo de la suma
        # para la capa de sigma y la capa de mu
        delta = (
                np.dot(gradient_mean, self.layers[self.latent_idx][0].weights) * self.derivative_activation_function(self.layers[self.latent_idx - 1].excitement)
                + np.dot(gradient_std, self.layers[self.latent_idx][1].weights) * self.derivative_activation_function(self.layers[self.latent_idx - 1].excitement)
        )

        delta_w.append(delta.reshape(-1, 1) @ np.transpose(self.layers[self.latent_idx-2].output.reshape(-1, 1)))
        prev_delta = delta

        for idx in range(self.latent_idx - 2, 0, -1):
            delta = np.dot(prev_delta, self.layers[idx + 1].weights) * self.derivative_activation_function(self.layers[idx].excitement)
            delta_w.append(delta.reshape(-1, 1) @ np.transpose(self.layers[idx - 1].output.reshape(-1, 1)))
            prev_delta = delta

        #  = = = = Calculamos el delta W de la capa inicial = = = =
        delta = np.dot(prev_delta, self.layers[1].weights) * self.derivative_activation_function(self.layers[0].excitement)
        delta_w.append(delta.reshape(-1, 1) @ np.transpose(self.input.reshape(-1, 1)))

        delta_w.reverse()

        return delta_w

    def update_all_weights(self, delta_w):
        for idx, layer in enumerate(self.layers):
            if idx == self.latent_idx:
                layer[0].update_weights(delta_w[idx][0])
                layer[1].update_weights(delta_w[idx][1])
            else:
                layer.update_weights(delta_w[idx])

    def train(self, limit, train_data):
        i = 0
        min_error = float("inf")
        while i < limit:
            error = 0

            for elem in train_data:
                result, mean, std = self.forward_propagation(elem)
                delta_w = self.backward_propagation(elem, result)
                self.update_all_weights(delta_w)

                error += total_loss(mean, std, result, elem)

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
            result, mean, std = self.forward_propagation(a)

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



