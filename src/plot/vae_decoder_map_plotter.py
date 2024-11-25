import numpy as np
import matplotlib.pyplot as plt
import pickle  # To load the .pkl file

# Load the previously generated .pkl file
with open("./output/output_data.pkl", "rb") as f:
    output_data = pickle.load(f)

decoded_images = output_data["images"]  # Extract the decoded images
coordinates = output_data["coordinates"]  # Extract the latent space coordinates

# Define the grid size (assumes a square grid)
grid_size = int(np.sqrt(len(decoded_images)))
assert grid_size ** 2 == len(decoded_images), "Data size must form a perfect square grid."

# Check the actual image size
image_size = int(np.sqrt(len(decoded_images[0])))  # Calculate the size from one image (assuming square images)

# Plot the decoded images in a grid
fig, ax = plt.subplots(grid_size, grid_size, figsize=(10, 10))

for i in range(grid_size):
    for j in range(grid_size):
        img_idx = i * grid_size + j
        image = decoded_images[img_idx]
        ax[i, j].imshow(np.array(image).reshape(image_size, image_size), cmap="gray")  # Use the correct shape
        ax[i, j].axis("off")  # Turn off axes for cleaner visualization

plt.subplots_adjust(wspace=0, hspace=0)
plt.show()
