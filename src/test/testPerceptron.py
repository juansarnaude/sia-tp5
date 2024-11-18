import json
import pandas as pd
import numpy as np

from src.models.MultiLayerPerceptron import MultiLayerPerceptron
from src.models.ActivationFunction import Tanh
from src.models.Optimizer import Adam, Momentum, GradientDescent

if __name__ == "__main__":

    # Load the data
    df = pd.read_csv('./input/perceptronTesting/xor.csv')
    X = df[['x1', 'x2']].values
    y = df['y'].values.reshape(-1, 1)

    # Create and configure the network
    activation = Tanh(input_range=(-3, 3), output_range=(-3, 3))
    optimizer = Adam(learning_rate=0.01)
    layer_sizes = [2, 4, 1]  # 2 inputs, 4 hidden neurons, 1 output

    # Create and train the network
    mlp = MultiLayerPerceptron(layer_sizes, activation, optimizer)
    mlp.train(X, y, epochs=1000, batch_size=1)

    # Make predictions
    predictions = mlp.predict(X)
    print("\nFinal predictions:")
    print("X1\tX2\tTarget\tPredicted")
    print("-" * 40)
    for i in range(len(X)):
        print(f"{X[i,0]:.1f}\t{X[i,1]:.1f}\t{y[i,0]:d}\t{predictions[i,0]:.2f}")

    # Calculate accuracy
    accuracy = np.mean((predictions >= 0) == (y >= 0)) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")

