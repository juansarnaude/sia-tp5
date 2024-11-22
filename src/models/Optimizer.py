import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):
    @abstractmethod
    def update(self, weights, gradients, layer_idx):
        pass

class GradientDescent(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, weights, gradients, layer_idx):
        return weights - self.learning_rate * gradients

class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}

    def update(self, weights, gradients, layer_idx):
        if layer_idx not in self.velocity:
            self.velocity[layer_idx] = np.zeros_like(weights)

        self.velocity[layer_idx] = self.momentum * self.velocity[layer_idx] - self.learning_rate * gradients
        return weights + self.velocity[layer_idx]

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate=learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 1
    
    def update(self, weights, gradients, layer_idx):
        if layer_idx not in self.m:
            self.m[layer_idx] = np.zeros_like(gradients)
            self.v[layer_idx] = np.zeros_like(gradients)
        
        self.m[layer_idx] = self.beta1 * self.m[layer_idx] + (1 - self.beta1) * gradients
        self.v[layer_idx] = self.beta2 * self.v[layer_idx] + (1 - self.beta2) * (gradients**2)
        
        m_hat = np.divide(self.m[layer_idx], (1 - self.beta1**self.t))
        v_hat = np.divide(self.v[layer_idx], (1 - self.beta2**self.t))
        
        self.t += 1
        return -self.learning_rate * np.divide(m_hat, (np.sqrt(v_hat) + self.epsilon))