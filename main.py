from perceptron.multi_perceptron import *


layer_config = [35, 25, 10, 2, 2, 10, 25, 35]

perceptron = MultiPerceptron(layer_config, theta_logistic, theta_logistic_derivative, 0.3, 0.3)

test = np.array([1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5])






