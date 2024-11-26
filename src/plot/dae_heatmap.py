import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.utils.noise import salt_and_pepper
from src.utils.font import parse_h_file_to_numpy, to_bin_array

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



# Function to plot a heatmap for each character
def plot_character_heatmaps(characters):
    num_characters = len(characters)

    # Calculate the grid dimensions (e.g., square grid, adjust as needed)
    num_cols = int(np.ceil(np.sqrt(num_characters)))  # Columns based on square root
    num_rows = int(np.ceil(num_characters / num_cols))  # Rows based on columns

    # Create the figure with a grid of subplots and extra space for the colorbar
    fig = plt.figure(figsize=(num_cols * 2, num_rows * 2))

    # Create a GridSpec to define the subplots and colorbar space
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(num_rows + 1, num_cols, figure=fig, height_ratios=[1] * num_rows + [0.05])

    # Create subplots for characters
    axes = [fig.add_subplot(gs[i // num_cols, i % num_cols]) for i in range(num_characters)]

    # Loop through the characters and plot each one
    for i, (character, ax) in enumerate(zip(characters, axes)):
        character_matrix = character.reshape(7, 5)  # Reshape each character to 7x5 matrix

        # Plot the heatmap using imshow
        cax = ax.imshow(character_matrix, cmap='Blues', interpolation='nearest', vmin=0, vmax=1)

        # Customize the plot
        ax.set_title(f"Character {i+1}")
        ax.set_xticks([])  # Remove x-tick labels
        ax.set_yticks([])  # Remove y-tick labels

    # Turn off any extra axes (in case there are unused subplots)
    for j in range(num_characters, len(axes)):
        axes[j].axis('off')

    # Add a colorbar below the plots to show the range of pixel values
    cbar_ax = fig.add_subplot(gs[-1, :])  # Use the last row for the colorbar
    fig.colorbar(cax, cax=cbar_ax, orientation='horizontal', label="Pixel Value")

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

# Example: Random 7x5 binary matrices (representing characters)
# You should replace this with your actual character dataset
# Example: 7x5 character matrix (binary values, 1 for filled, 0 for empty)
characters = parse_input("./input/font.h")
#characters = get_noisy_dataset(X)

# csv_file = "./output/characters_matrix_dae.csv"  # Replace this with the actual path to your CSV file
#
# # Read the CSV into a pandas DataFrame
# df = pd.read_csv(csv_file, header=None)
#
# # Convert the dataframe to a numpy array
# data = df.to_numpy()
#
# # Check the shape of the loaded data
# print("Data shape:", data.shape)  # Ensure the data has the correct dimensions (multiple of 35 rows)
#
# # Reshape the data into a 7x5 grid per character
# # Assuming each character is represented by 7 rows of 5 columns
# num_rows = 7
# num_cols = 5

# Reshaping: Each character will have 7x5 = 35 elements
#characters = data.reshape(-1, num_rows, num_cols)

# Plot the heatmaps for the characters
plot_character_heatmaps(characters)
