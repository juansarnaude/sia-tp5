import numpy as np

from src.models.ActivationFunction import ActivationFunction
from src.models.Layer import Layer
from src.models.Optimizer import Optimizer

def mse(x,y):
    return np.mean(np.power(x-y, 2))

def err(x,y):
    return x-y

class MultiLayerPerceptron:
    def __init__(self, layers: [], activation_function: ActivationFunction, optimizer: Optimizer):
        self.activation_function = activation_function
        self.optimizer = optimizer
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(Layer(layers[i], layers[i + 1], self.activation_function))

    def feed_forward(self, output):
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backpropagation(self, error):
        for i in range(len(self.layers) - 1, -1, -1):
            gradient, error = self.layers[i].backwards(error)
            weight_update = np.dot(self.layers[i].inputs.T, gradient)
            self.layers[i].neurons = self.optimizer.update(self.layers[i].neurons , weight_update, i)
        return error


    def train(self, inputs, targets, epochs):
        input_length = len(inputs)

        for x in range(epochs):
            total_error = 0
            for i in range(input_length):
                output = self.feed_forward(inputs[i])
                total_error += mse(output,targets[i])
                error = err(targets[i], output)
                self.backpropagation(error)

            total_error /= input_length

            if x % 100:
                print("Epoch: " + str(x) + " error: " + str(total_error))

    def predict(self, x):
        return self.feed_forward(x)
