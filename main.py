import random
import time

from perceptron.functions import *
from perceptron.multi_perceptron import *
from perceptron.optimizers import ADAM
from training_data.font import *


if __name__ == "__main__":



    random.seed(2)
    np.random.seed(2)

    layer_config = [35, 30, 30, 20, 20, 10, 2, 10, 20, 20, 30, 30, 35]
    beta = 1

    optimizer = ADAM
    optimizer_args = [0.001, 0.9, 0.999, 1e-8]

    batch_size = len(fonts)

    autoencoder = MultiPerceptron(layer_config, partial(sigmoid, beta), partial(sigmoid_derivative, beta), optimizer, optimizer_args)

    t1 = time.time()
    min_error = autoencoder.train(50000, fonts, fonts, batch_size)
    t2 = time.time()

    print(min_error, t2-t1)
    

    autoencoder.test(fonts, fonts)



