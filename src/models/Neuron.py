import numpy as np

class Neuron:
    np.random.seed(3) # Do not touch, good starting values with this seed

    def __init__(self, input_size):
        self.weights = np.random.randn(input_size) * 0.1
        self.bias = np.random.randn() * 0.1
    
    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.bias