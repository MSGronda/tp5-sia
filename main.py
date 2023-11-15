from perceptron.multi_perceptron import *
from training_data.font import *


if __name__ == "__main__":

    layer_config = [35, 20, 10, 2, 10, 20, 35]
    learning_constant = 0.01
    beta = 2

    autoencoder = MultiPerceptron(layer_config, theta_logistic, theta_logistic_derivative, learning_constant, beta)

    t1 = time.time()
    min_error = autoencoder.train(2000, fonts, fonts)
    t2 = time.time()

    print(min_error, t2-t1)

    autoencoder.test(fonts, fonts)



