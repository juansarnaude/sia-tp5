from abc import ABC, abstractmethod
import numpy as np

class ActivationFunction(ABC):
    def __init__(self, input_range=None, output_range=None):
        self.input_range = input_range
        self.output_range = output_range
    
    def normalize(self, x):
        if self.input_range is None:
            return x
        input_min, input_max = self.input_range
        return (x - input_min) / (input_max - input_min) * 2 - 1
    
    def denormalize(self, x):
        if self.output_range is None:
            return x
        output_min, output_max = self.output_range
        return (x + 1) / 2 * (output_max - output_min) + output_min

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass

    def __call__(self, x, derivative=False):
        if derivative:
            return self.derivative(x)
        return self.forward(x)

class Tanh(ActivationFunction):
    def forward(self, x):
        x_norm = self.normalize(x)
        result = np.tanh(x_norm)
        return self.denormalize(result)
    
    def derivative(self, x):
        x_norm = self.normalize(x)
        result = 1 - np.tanh(x_norm)**2
        if self.output_range:
            output_min, output_max = self.output_range
            result *= (output_max - output_min)
        return result

class Sigmoid(ActivationFunction):
    def forward(self, x):
        # Normalize the input and apply the sigmoid function
        x_norm = self.normalize(x)
        result = 1 / (1 + np.exp(-x_norm))  # Sigmoid function
        return self.denormalize(result)

    def derivative(self, x):
        # Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
        x_norm = self.normalize(x)
        sigmoid_output = 1 / (1 + np.exp(-x_norm))
        result = sigmoid_output * (1 - sigmoid_output)

        # Adjust for the output range
        if self.output_range:
            output_min, output_max = self.output_range
            result *= (output_max - output_min)
        return result