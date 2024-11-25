import numpy as np
from matplotlib import pyplot as plt

from src.autoencoders.autoencoder import parse_input
from src.autoencoders.variationalAutoencoder import VariationalAutoencoder
from src.models.Optimizer import Adam
from src.models.ActivationFunction import Tanh
from src.autoencoders.emojis import emoji_images,emoji_size,emoji_chars,emoji_names,vector_to_emoji

# Example input data (replace with your actual data)
X = parse_input("./input/font.h")

# Example input data (replace with your actual data)
emoji_indexes = np.array([0, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 23, 24, 26,
                          28, 29, 31, 32, 33, 35, 36, 38, 39, 41, 46, 48, 50, 51, 54, 55,
                          57, 58, 59, 61, 62, 63, 64, 65, 67, 73, 75, 78, 81, 83, 84, 85,
                          90, 91, 92, 93, 95])

data = np.array(emoji_images)
dataset_input = data[emoji_indexes]
dataset_input_list = list(dataset_input)

# Hyperparameters
latent_dim = 2
encoder_layers = [400,300,200,20]
decoder_layers = [20,200,300,400]
activation = Tanh(input_range=(0, 1), output_range=(0, 1))
optimizer1 = Adam(learning_rate=0.01)
optimizer2 = Adam(learning_rate=0.01)

# Instantiate the model
vae = VariationalAutoencoder(
    latent_dim=latent_dim,
    encoder_layers=encoder_layers,
    decoder_layers=decoder_layers,
    activation=activation,
    optimizer1=optimizer1,
    optimizer2=optimizer2,
)

# Train the model
vae.train(dataset_input_list, epochs=100, batch_size=len(dataset_input_list))

print("Finished Execution")
print("-------------------------")

coordinates=[]
for emoji in dataset_input_list:
    coordinates.append(vae.encode(emoji))

print(coordinates)

with open(f"./output/latent_predictions_vae_emojis.csv", "w") as file:
    file.write("x,y\n")
    for latent_prediction in coordinates:
        file.write(",".join(map(str, latent_prediction)) + "\n")

# Generate new data

# z_sample = np.random.rand(latent_dim)  # Random latent vector
# generated_sample = vae.generate(z_sample)
# vector_to_emoji(generated_sample)



# # Encode some input data
# for i in range(10):# Encode the first 10 samples
#     mu, log_var = vae.encode(dataset_input_list[i])
#     print("X:",dataset_input_list[i])
#     print("Latent Space Mean:", mu)
#     print("Latent Space Log Variance:", log_var)
#
# # Decode a latent vector
# decoded_sample = vae.decode(z_sample)
# print("Decoded Sample:", decoded_sample)
