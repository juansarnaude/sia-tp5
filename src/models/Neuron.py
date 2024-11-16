import numpy as np

class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size) * 0.1
        self.bias = np.random.randn() * 0.1
    
    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.bias