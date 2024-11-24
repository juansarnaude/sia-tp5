
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

    def backpropagation_vae(self, layer_output_error):
        # Initialize accumulators for weight and bias gradients
        total_weight_gradients = [np.zeros_like(layer.get_weights()) for layer in self.layers]
        total_bias_gradients = [np.zeros_like(layer.get_biases()) for layer in self.layers]

        # Process each error in the batch
        for error in layer_output_error:
            delta = error
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

            for i in range(len(self.layers)):
                total_weight_gradients[i] += weight_gradients[i]
                total_bias_gradients[i] += bias_gradients[i]

        total_weight_gradients = [wg / len(layer_output_error) for wg in total_weight_gradients]
        total_bias_gradients = [bg / len(layer_output_error) for bg in total_bias_gradients]

        return total_weight_gradients, total_bias_gradients

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
        with open(f"./output/total_loss_per_epoch.csv", "w") as file:
            file.write("epoch,average_loss\n")

            for epoch in range(epochs):
                total_loss = 0

                indices = np.arange(len(X))
                np.random.shuffle(indices)
                X, y = X[indices], y[indices]

                for i in range(0, len(X), batch_size):
                    batch_X = X[i:i + batch_size]
                    batch_y = y[i:i + batch_size]

                    batch_weight_gradients = [np.zeros_like(layer.get_weights()) for layer in self.layers]
                    batch_bias_gradients = [np.zeros_like(layer.get_biases()) for layer in self.layers]

                    for x_sample, y_sample in zip(batch_X, batch_y):
                        weight_gradients, bias_gradients = self.backpropagation(x_sample, y_sample)
                        for layer_idx, (w_grad, b_grad) in enumerate(zip(weight_gradients, bias_gradients)):
                            batch_weight_gradients[layer_idx] += w_grad
                            batch_bias_gradients[layer_idx] += b_grad

                    for layer_idx in range(len(self.layers)):
                        batch_weight_gradients[layer_idx] /= len(batch_X)
                        batch_bias_gradients[layer_idx] /= len(batch_X)

                    for layer_idx, layer in enumerate(self.layers):
                        new_weights = self.optimizer.update(layer.get_weights(), batch_weight_gradients[layer_idx],
                                                            f"w{layer_idx}")
                        new_biases = self.optimizer.update(layer.get_biases(), batch_bias_gradients[layer_idx],
                                                           f"b{layer_idx}")
                        layer.update_weights(new_weights)
                        layer.update_biases(new_biases)


                if epoch % 100 == 0:
                    predictions = np.array([self.feed_forward(x_sample) for x_sample in X])
                    total_loss = np.mean([self.mse(y_true, y_pred) for y_true, y_pred in zip(y, predictions)])
                    print(f"Epoch {epoch}, Average Loss: {total_loss}")
                    file.write(f"{epoch},{total_loss}\n")

    def update_weights(self, gradients, epoch):
        weight_gradients, bias_gradients = gradients

        for layer_idx, layer in enumerate(self.layers):
            new_weights = self.optimizer.update(
                layer.get_weights(),
                weight_gradients[layer_idx],
                f"weights_layer_{layer_idx}",
                epoch=epoch
            )
            layer.update_weights(new_weights)

            new_biases = self.optimizer.update(
                layer.get_biases(),
                bias_gradients[layer_idx],
                f"biases_layer_{layer_idx}",
                epoch=epoch
            )
            layer.update_biases(new_biases)

    def predict(self, x):
        return self.feed_forward(x)
    
    @staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)