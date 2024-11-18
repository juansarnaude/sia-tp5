import json
import pandas as pd
import numpy as np

from src.models.MultiLayerPerceptron import MultiLayerPerceptron
from src.models.ActivationFunction import Tanh
from src.models.Optimizer import Adam, Momentum, GradientDescent
from src.utils.font import parse_h_file_to_numpy, to_bin_array

def parse_input(input_file_path: str):
    hex_characters = parse_h_file_to_numpy(input_file_path)
    characters = np.empty((0, 35))

    for character in hex_characters:
        characters = np.vstack([characters, np.array(to_bin_array(character)).flatten()])

    return characters
if __name__ == "__main__":
    X = parse_input("./input/font.h")

     # Create and configure the network
    activation = Tanh(input_range=(0, 1), output_range=(0, 1))
    optimizer = Adam(learning_rate=0.01)
    layer_sizes = [35, 20, 10, 8, 4, 2] # This one does not work
    layers = layer_sizes + layer_sizes[::-1]

    # Create and train the network
    mlp = MultiLayerPerceptron(layers, activation, optimizer)
    mlp.train(X, X, epochs=1000, batch_size=1)

    # Make predictions
    predictions = mlp.predict(X[0])
    print(predictions)
    print(X[0])