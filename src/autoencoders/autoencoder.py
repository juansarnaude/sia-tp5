import json
import pandas as pd
import numpy as np

from src.models.MultiLayerPerceptron import MultiLayerPerceptron
from src.models.ActivationFunction import Tanh,Sigmoid
from src.models.Optimizer import Adam, Momentum, GradientDescent
from src.utils.font import parse_h_file_to_numpy, to_bin_array
from src.utils.normalize import zeroOrOne

def incorrect_pixel_count(input, predicted):
    count = 0
    for input_pixel, predicted_pixel in zip(input, predicted):
        if abs(input_pixel - predicted_pixel) > 0.5:
            count += 1
    return count

def parse_input(input_file_path: str):
    hex_characters = parse_h_file_to_numpy(input_file_path)
    characters = np.empty((0, 35))

    for character in hex_characters:
        characters = np.vstack([characters, np.array(to_bin_array(character)).flatten()])

    return characters
if __name__ == "__main__":
    X = parse_input("./input/font.h")

    # Create and configure the network
    # Adjusted autoencoder configuration with an input shape of 35
    activation = Tanh(input_range=(0, 1), output_range=(0, 1))  # Tanh works well if your data is normalized between -1 and 1
    optimizer = Adam(learning_rate=0.0005)  # Lower learning rate for stability

    # Layer sizes based on input shape of 35
    layer_sizes = [35, 20, 2] # Example layer sizes, decreasing towards the bottleneck layer (latent representation)
    layers = layer_sizes + layer_sizes[::-1][1:]  # Symmetric architecture, avoid duplicating the bottleneck layer

    # Create and train the autoencoder
    mlp = MultiLayerPerceptron(layers, activation, optimizer)
    mlp.train(X, X, epochs=3000, batch_size=32)  # Train with batch size of 32 for stability and efficiency

    predictions = []
    total_pixel_loss = 0

    for input in X:
        predictions.append(zeroOrOne(mlp.predict(input)))

    for prediction, input in zip(predictions, X):
        print('-'*40)
        print('prediction')
        print(prediction)
        print('input')
        print(input)

        for position, x in zip(prediction, input):
            if position != x:
                total_pixel_loss += 1

    print(total_pixel_loss)