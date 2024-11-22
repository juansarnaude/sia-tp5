import numpy as np

class Neuron:
    np.random.seed(3) # Do not touch, good starting values with this seed

    def __init__(self, input_size):
        limit = np.sqrt(6 / input_size)
        self.weights = np.random.uniform(0, limit, input_size)

        self.bias = 0.0

    
    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.bias