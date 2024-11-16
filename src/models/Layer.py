from src.models.Neuron import Neuron
from src.models.ActivationFunction import ActivationFunction

import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation_function: ActivationFunction):
        self.neurons = [Neuron(input_size) for _ in range(output_size)]
        self.activation_function = activation_function
        self.inputs = None
        self.outputs = None
        
    def forward(self, inputs):
        self.inputs = inputs
        neuron_outputs = np.array([neuron.forward(inputs) for neuron in self.neurons])
        self.outputs = self.activation_function(neuron_outputs)
        return self.outputs
    
    def get_weights(self):
        return np.array([neuron.weights for neuron in self.neurons])
    
    def get_biases(self):
        return np.array([neuron.bias for neuron in self.neurons])
    
    def update_weights(self, new_weights):
        for neuron, weights in zip(self.neurons, new_weights):
            neuron.weights = weights
    
    def update_biases(self, new_biases):
        for neuron, bias in zip(self.neurons, new_biases):
            neuron.bias = bias