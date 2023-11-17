from abc import abstractmethod
import numpy as np


class Optimizer:
    @abstractmethod
    def calc_delta_w(self, delta_w):
        raise RuntimeError("Implement me!")


class Momentum(Optimizer):
    def __init__(self, learning_rate=0.8):
        # Variable para optimizacion Momentum
        self.learning_rate = learning_rate
        self.prev_delta = 0

    def calc_delta_w(self, delta_w):
        new_delta = delta_w + self.learning_rate * self.prev_delta
        self.prev_delta = new_delta
        return new_delta


class ADAM(Optimizer):
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):

        # Variables para optimizacion ADAM
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m_t = 0
        self.v_t = 0
        self.t = 0

    def calc_delta_w(self, delta_w):
        self.t += 1

        self.m_t = self.beta1 * self.m_t + (1 - self.beta1) * delta_w
        self.v_t = self.beta2 * self.v_t + (1 - self.beta2) * np.power(delta_w, 2)

        final_m_t = self.m_t / (1 - self.beta1 ** self.t)
        final_v_t = self.v_t / (1 - self.beta2 ** self.t)

        return self.alpha * final_m_t / (np.sqrt(final_v_t) + self.epsilon)

