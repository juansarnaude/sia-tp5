import json
import pandas as pd
import numpy as np

from src.models.MultiLayerPerceptron import MultiLayerPerceptron
from src.models.ActivationFunction import Tanh, Sigmoid
from src.models.Optimizer import Adam
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
    characters = np.empty((0, 35))  # 35 bits per character (e.g., flattened 7x5 grid)

    for character in hex_characters:
        characters = np.vstack([characters, np.array(to_bin_array(character)).flatten()])

    return characters

def generate_batches(data, batch_size):
    n_samples = data.shape[0]
    n_batches = n_samples // batch_size
    batches = np.array_split(data, n_batches)
    return batches

if __name__ == "__main__":
    X = parse_input("./input/font.h")

    activation = Tanh(input_range=(0, 1), output_range=(0, 1))
    optimizer = Adam(learning_rate=0.0001)
    layers = [35, 10, 2, 10, 35]

    mlp = MultiLayerPerceptron(layers, activation, optimizer)

    batch_size = 10
    training_batches = generate_batches(X, batch_size)

    mlp.train(training_batches, training_batches, 3000)

    # Test the model
    predictions = []
    total_pixel_loss = 0

    # for input in X:
    #     predicted = zeroOrOne(mlp.predict(input))
    #     predictions.append(predicted)
    #
    #     for predicted_pixel, input_pixel in zip(predicted, input):
    #         if predicted_pixel != input_pixel:
    #             total_pixel_loss += 1

    # for prediction, input in zip(predictions, X):
    #     print('-' * 40)
    #     print('Prediction:')
    #     print(prediction)
    #     print('Input:')
    #     print(input)

    print("--------------------")
    print("Total pixels: ", len(X)*len(X[0]) , " Correct: ", len(X)*len(X[0])-total_pixel_loss, " Incorrect: ", total_pixel_loss, " Error %: ", 100*total_pixel_loss/(len(X)*len(X[0])))