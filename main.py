import time

from perceptron.functions import sigmoid, sigmoid_derivative
from perceptron.multi_perceptron import *
from perceptron.optimizers import ADAM
from training_data.font import *


if __name__ == "__main__":

    layer_config = [35, 25, 15, 7, 2, 7, 15, 25, 35]
    beta = 2

    optimizer = ADAM
    optimizer_args = [0.001, 0.9, 0.999, 1e-8]

    autoencoder = MultiPerceptron(layer_config, partial(sigmoid, beta), partial(sigmoid_derivative, beta), optimizer, optimizer_args)

    t1 = time.time()
    min_error = autoencoder.train(1000, fonts, fonts)
    t2 = time.time()

    print(min_error, t2-t1)

    autoencoder.test(fonts, fonts)



