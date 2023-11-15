from perceptron.multi_perceptron import *
from training_data.font import *


if __name__ == "__main__":

    layer_config = [35, 25, 10, 2, 10, 25, 35]
    learning_constant = 0.3
    beta = 0.3

    autoencoder = MultiPerceptron(layer_config, theta_logistic, theta_logistic_derivative, learning_constant, beta)

    min_error = autoencoder.train(1000, 0.1, fonts, fonts)

    print(min_error)

