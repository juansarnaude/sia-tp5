from src.models.Layer import Layer
from src.models.ActivationFunction import ActivationFunction

import numpy as np

class MultiLayerPerceptron:
    def __init__(self, layer_sizes, activation_function: ActivationFunction, optimizer):
        self.activation_function = activation_function
        self.optimizer = optimizer
        self.layers = []
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1], self.activation_function))
    
    def feed_forward(self, inputs):
        current_output = inputs
        for layer in self.layers:
            current_output = layer.forward(current_output)
        return current_output
    
    def backpropagation(self, inputs, targets):
        outputs = self.feed_forward(inputs)
        output_error = outputs - targets
        delta = output_error * self.activation_function(outputs, derivative=True)
        
        weight_gradients = []
        bias_gradients = []
        
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            
            weight_grad = np.outer(delta, layer.inputs)
            bias_grad = delta
            
            weight_gradients.insert(0, weight_grad)
            bias_gradients.insert(0, bias_grad)
            
            if i > 0:
                weights = layer.get_weights()
                delta = np.dot(weights.T, delta) * self.activation_function(layer.inputs, derivative=True)
        
        return weight_gradients, bias_gradients
    
    def train(self, X, y, epochs=1000, batch_size=32):
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i + batch_size]
                batch_y = y[i:i + batch_size]
                
                for x_sample, y_sample in zip(batch_X, batch_y):
                    weight_gradients, bias_gradients = self.backpropagation(x_sample, y_sample)
                    
                    for layer_idx, (layer, weight_grad, bias_grad) in enumerate(zip(self.layers, weight_gradients, bias_gradients)):
                        new_weights = self.optimizer.update(layer.get_weights(), weight_grad, f"w{layer_idx}")
                        new_biases = self.optimizer.update(layer.get_biases(), bias_grad, f"b{layer_idx}")
                        
                        layer.update_weights(new_weights)
                        layer.update_biases(new_biases)
                    
                    prediction = self.feed_forward(x_sample)
                    total_loss += self.mse(y_sample, prediction)
            
            if epoch % 100 == 0:
                avg_loss = total_loss / len(X)
                print(f"Epoch {epoch}, Average Loss: {avg_loss}")
    
    def predict(self, X):
        return np.array([self.feed_forward(x) for x in X])
    
    @staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)