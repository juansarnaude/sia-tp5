import numpy as np

from src.models.ActivationFunction import ActivationFunction
from src.models.Layer import Layer
from src.models.Optimizer import Optimizer

def mse(x,y):
    return np.mean(np.power(x-y, 2))

def err(x,y):
    return y-x

class MultiLayerPerceptron:
    def __init__(self, layers: [], activation_function: ActivationFunction, optimizer: Optimizer):
        self.activation_function = activation_function
        self.optimizer = optimizer
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(Layer(layers[i], layers[i + 1], self.activation_function))

    def feed_forward(self, output):
        aux = output
        for layer in self.layers:
            aux = layer.forward(aux)
        return aux

    def backpropagation(self, error, index):
        gradient, back_error = self.layers[index].backwards(error)
        weight_update = np.dot(self.layers[index].inputs.T, gradient)
        self.layers[index].neurons += self.optimizer.update(self.layers[index].neurons , weight_update, index)
        return back_error


    def train(self, inputs, targets, epochs):
        input_length = len(inputs)

        for x in range(epochs):
            total_error = 0
            for i in range(input_length):
                output = inputs[i]
                output = self.feed_forward(output)
                total_error += mse(targets[i],output)
                error = err(targets[i], output)
                for j in range(len(self.layers) - 1, -1, -1):
                    error = self.backpropagation(error,j)

            total_error /= input_length
            computed_error = self.compute_error(inputs, targets)

            if x % 100:
                print("Epoch: " + str(x) + " error: " + str(total_error) + " Computed Error: " + str(computed_error))

            if computed_error == 0:
                break

    def predict(self, x):
        samples = len(x)
        result = []
        for i in range(samples):
            output = x[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)
        return result


    def compute_error(self, dataset, expected):
        to_return = 0
        result = self.predict(dataset)[0]
        expected = expected[0]
        for i in range(0, len(result)):
            result[i] = result[i].round().astype(int)

            errors = 0
            for j in range(0, len(result[i])):
                errors += np.where(result[i][j]!=expected[i][j],1,0)

            if errors > to_return:
                to_return = errors

        return to_return