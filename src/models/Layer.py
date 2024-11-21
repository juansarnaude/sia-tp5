from src.models.ActivationFunction import ActivationFunction

import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation_function: ActivationFunction):
        self.activation_function = activation_function
        self.neurons = np.random.uniform(low=0, high=1, size=(input_size, output_size)) #TODO make the initialization parametrizable
        self.inputs = None
        self.output_plain = None
        self.outputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output_plain = np.dot(self.inputs, self.neurons)
        self.outputs = self.activation_function(self.output_plain)
        return self.outputs

    def backwards(self, error):
        gradient = np.multiply(self.activation_function.derivative(self.output_plain), error)
        back_error = np.dot(gradient,self.neurons.T)
        return gradient, back_error

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