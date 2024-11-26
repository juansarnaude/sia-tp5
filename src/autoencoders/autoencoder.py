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

def get_encoder(mlp, layer_sizes):
    encoder_layers = layer_sizes[:2 + 1]  # Hasta la capa de 2 neuronas (espacio latente)
    encoder = MultiLayerPerceptron(encoder_layers, mlp.activation_function, mlp.optimizer)
    encoder.layers = mlp.layers[:len(encoder_layers) - 1]
    return encoder

def get_decoder(mlp, layer_sizes):
    decoder_layers = layer_sizes[2:]  # Desde la capa de 2 neuronas en adelante
    decoder = MultiLayerPerceptron(decoder_layers, mlp.activation_function, mlp.optimizer)
    decoder.layers = mlp.layers[len(decoder_layers) - 1:]
    return decoder

if __name__ == "__main__":

    with open("./config/config_autoencoder.json", 'r') as file:
        config = json.load(file)

        n = config["learning_rate"]
        inner_layers = config["layers"]
        epochs = config["epochs"]
        output_file_name = config["output_file_name"]
        optimizer_str = config["optimizer"]
        momentum_weight = config["momentum_weight"]

        X = parse_input("./input/font.h")

        # Create and configure the network
        # Adjusted autoencoder configuration with an input shape of 35
        activation = Tanh(input_range=(0, 1), output_range=(0, 1))  # Tanh works well if your data is normalized between -1 and 1
        
        optimizer = None

        if optimizer_str == "adam":
            optimizer = Adam(learning_rate=n)  # Lower learning rate for stability
        elif optimizer_str == "momentum":
            optimizer = Momentum(learning_rate=n, momentum=momentum_weight)
        elif optimizer_str == "gradient_descent":
            optimizer = GradientDescent(learning_rate=n)

        # Layer sizes based on input shape of 35
        layer_sizes = [35] + inner_layers + [2]
        layers = layer_sizes + layer_sizes[::-1][1:]  # Symmetric architecture, avoid duplicating the bottleneck layer

        # Create and train the autoencoder
        mlp = MultiLayerPerceptron(layers, activation, optimizer, output_file_name)
        mlp.train(X, X, epochs=epochs, batch_size=32)  # Train with batch size of 32 for stability and efficiency

        predictions = []
        total_pixel_loss = 0

        for input in X:
            predictions.append(zeroOrOne(mlp.predict(input)))

        for prediction, input in zip(predictions, X):
            pixel_loss = 0

            for position, x in zip(prediction, input):
                if position != x:
                    pixel_loss += 1

            total_pixel_loss += pixel_loss
            print(pixel_loss)

        #Predictions
        predictions_matrix = [np.reshape(array, (7, 5)) for array in predictions]
        with open(f"./output/characters_matrix_autoencoder.csv", "w") as file:
            for matrix in predictions_matrix:
                # Escribir cada fila de la matriz en el archivo CSV
                for row in matrix:
                    file.write(",".join(f"{value}" for value in row) + "\n")

        #EJ3 Vemos las coordenadas del encoder
        encoder=get_encoder(mlp,layers)

        latent_predictions = []
        for input in X:
            latent_predictions.append(encoder.predict(input))

        with open(f"./output/latent_predictions_autoencoder.csv", "w") as file:
            file.write("x,y\n")
            for latent_prediction in latent_predictions:
                file.write(",".join(map(str, latent_prediction)) + "\n")

        #Creamos un mapa de representaciones del espacio latente
        decoder=get_decoder(mlp,layers)

        decoder_predictions=[]

        sample_size = 15
        array = np.linspace(0, 1, sample_size)
        coordinates = [[[i, j] for j in array] for i in array[::-1]]

        for i in range(sample_size):
            for j in range(sample_size):
                decoder_predictions.append(decoder.predict(coordinates[i][j]))

        decoder_predictions_matrix = [np.reshape(array, (7, 5)) for array in decoder_predictions]

        with open(f"./output/new_characters_autoencoder.csv", "w") as file:
            for matrix in decoder_predictions_matrix:
                # Escribir cada fila de la matriz en el archivo CSV
                for row in matrix:
                    file.write(",".join(f"{value}" for value in row) + "\n")

        #Ej4 generate new characters
        decoder_new_predictions=[]

        x_y_coordinates=[[0,0.72],[0.77,0.41],[0.75,0.4]]
        for x_y in x_y_coordinates:
            decoder_new_predictions.append(zeroOrOne(decoder.predict(x_y)))

        decoder_new = [np.reshape(array, (7, 5)) for array in decoder_new_predictions]

        with open(f"./output/new_characters.csv", "w") as file:
            for matrix in decoder_new:
                # Escribir cada fila de la matriz en el archivo CSV
                for row in matrix:
                    file.write(",".join(f"{value}" for value in row) + "\n")

        print("Total pixels: ", len(X)*len(X[0]) , " Correct: ", len(X)*len(X[0])-total_pixel_loss, " Incorrect: ",
            total_pixel_loss, " Error %: ", 100*total_pixel_loss/(len(X)*len(X[0])))