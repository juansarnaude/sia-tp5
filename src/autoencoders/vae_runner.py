import csv

import numpy as np
from matplotlib import pyplot as plt

from src.autoencoders.autoencoder import parse_input
from src.autoencoders.variationalAutoencoder import VariationalAutoencoder
from src.models.Optimizer import Adam
from src.models.ActivationFunction import Tanh
from src.autoencoders.emojis import emoji_images,vector_to_emoji,NUMBER_OF_EMOJIS
import pickle

# Example input data (replace with your actual data)
X = parse_input("./input/font.h")

# Example input data (replace with your actual data)
emoji_indexes = np.arange(0,NUMBER_OF_EMOJIS)
data = np.array(emoji_images)
dataset_input = data[emoji_indexes]
dataset_input_list = list(dataset_input)

# Hyperparameters
latent_dim = 2
encoder_layers = [400,200]
decoder_layers = [200,400]
activation = Tanh(input_range=(0, 1), output_range=(0, 1))
optimizer1 = Adam(learning_rate=0.001)
optimizer2 = Adam(learning_rate=0.001)

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
_, total_loss = vae.train(dataset_input_list, epochs=1000, batch_size=1)

print("Finished Execution")
print("-------------------------")

# coordinates=[]
# for emoji in dataset_input_list:
#     coordinates.append(vae.encode(emoji))
#
#
# with open(f"./output/latent_predictions_vae_emojis.csv", "w") as file:
#     file.write("x,y\n")
#     for latent_prediction in coordinates:
#         file.write(",".join(map(str, latent_prediction)) + "\n")
#
#     for i in range(0,len(dataset_input_list)):
# #     vector_to_emoji(dataset_input_list[i])
#         vector_to_emoji(vae.generate(vae.encode(dataset_input_list[i])))

for da in dataset_input_list:
    print(vae.encode(da))

emoji_img_list = []
emoji_img_coords_list = []

range_min, range_max = -1, 1
grid_size = 20
step_size = (range_max - range_min) / grid_size

for x in np.arange(range_min, range_max, step_size):
    for y in np.arange(range_min, range_max, step_size):
        z = [x, y]
        decoded_image = vae.decode(z)
        emoji_img_list.append(decoded_image)
        emoji_img_coords_list.append(z)


output_data = {
    "images": emoji_img_list,
    "coordinates": emoji_img_coords_list,
}

output_file = "./output/output_data.pkl"
with open(output_file, "wb") as f:
    pickle.dump(output_data, f)

print(f"Data saved to {output_file}")

with open("./output/vae_total_loss_through_epochs.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["epoch", "total_loss"])
    for epoch, t_l in enumerate(total_loss):
        writer.writerow([epoch, t_l])

print(f"Data has been written to { './output/vae_total_loss_through_epochs' }")





# # Generate new data
#
# z_sample = np.random.rand(latent_dim)  # Random latent vector
# generated_sample = vae.generate(z_sample)
# print("Generated Sample:", generated_sample)
#
# # Encode some input data
# for i in range(10):# Encode the first 10 samples
#     mu, log_var = vae.encode(X[i])
#     print("X:",X[i])
#     print("Latent Space Mean:", mu)
#     print("Latent Space Log Variance:", log_var)
#
# # Decode a latent vector
# decoded_sample = vae.decode(z_sample)
# print("Decoded Sample:", decoded_sample)


