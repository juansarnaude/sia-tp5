import json
import pandas as pd
import numpy as np
import copy
import csv

from src.models.MultiLayerPerceptron import MultiLayerPerceptron
from src.models.ActivationFunction import Tanh,Sigmoid
from src.models.Optimizer import Adam, Momentum, GradientDescent
from src.utils.font import parse_h_file_to_numpy, to_bin_array
from src.utils.normalize import zeroOrOne
from src.utils.noise import salt_and_pepper

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

def get_noisy_dataset(dataset, noise_level=0.1):
    # Validate the input
    if dataset.ndim != 2 or dataset.shape[1] != 35:
        raise ValueError("The dataset must be a 2D array where each sample has 35 elements.")
    
    noisy_dataset = np.array([
        salt_and_pepper(sample, noise_level) for sample in dataset
    ])
    
    return noisy_dataset


if __name__ == "__main__":
    X = parse_input("./input/font.h")
    testing_set = get_noisy_dataset(X, 0.1)

    noisy_dataset = [np.reshape(array, (7, 5)) for array in testing_set]

    # Write the matrices to the CSV file
    with open("./output/noisy_dataset_sp01.csv", mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for character in noisy_dataset:
            # Write each row of the 7x5 matrix
            writer.writerows(character)

    # Create and configure the network
    # Adjusted autoencoder configuration with an input shape of 35
    activation = Tanh(input_range=(0, 1), output_range=(0, 1))  # Tanh works well if your data is normalized between -1 and 1
    optimizer = Adam(learning_rate=0.001)  # Lower learning rate for stability

    # Layer sizes based on input shape of 35
    layer_sizes = [35, 32, 8, 2]
    layers = layer_sizes + layer_sizes[::-1][1:]  # Symmetric architecture, avoid duplicating the bottleneck layer

    # Create and train the autoencoder
    mlp = MultiLayerPerceptron(layers, activation, optimizer, "mse_vs_epoch_02_02")
    mlp.train_dae(X, testing_set, epochs=13000, batch_size=32)  # Train with batch size of 32 for stability and efficiency

    predictions = []
    total_pixel_loss = 0

    for noisy_input, expected in zip(testing_set, X):
        predictions.append(zeroOrOne(mlp.predict(noisy_input)))

    for prediction, input in zip(predictions, X):
        pixel_loss = 0

        for position, x in zip(prediction, input):
            if position != x:
                pixel_loss += 1

        total_pixel_loss += pixel_loss
        print(pixel_loss)

    predictions_matrix = [np.reshape(array, (7, 5)) for array in predictions]
    with open(f"./output/characters_matrix_dae.csv", "w") as file:
        for matrix in predictions_matrix:
            # Escribir cada fila de la matriz en el archivo CSV
            for row in matrix:
                file.write(",".join(f"{value}" for value in row) + "\n")

    print("--------------------")
    print("Total pixels: ", len(X)*len(X[0]) , " Correct: ", len(X)*len(X[0])-total_pixel_loss, " Incorrect: ",
          total_pixel_loss, " Error %: ", 100*total_pixel_loss/(len(X)*len(X[0])))