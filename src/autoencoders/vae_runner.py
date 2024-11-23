import numpy as np

from src.autoencoders.autoencoder import parse_input
from src.autoencoders.variationalAutoencoder import VariationalAutoencoder
from src.models.Optimizer import Adam
from src.models.ActivationFunction import Tanh

# Example input data (replace with your actual data)
X = parse_input("./input/font.h")

# Hyperparameters
latent_dim = 2
encoder_layers = [35, 30]
decoder_layers = [30, 35]
activation = Tanh(input_range=(0, 1), output_range=(0, 1))
optimizer = Adam(learning_rate=0.001)

# Instantiate the model
vae = VariationalAutoencoder(
    latent_dim=latent_dim,
    encoder_layers=encoder_layers,
    decoder_layers=decoder_layers,
    activation=activation,
    optimizer=optimizer,
)

# Train the model
vae.train(X, epochs=1000, batch_size=32)

# Generate new data
z_sample = np.random.rand(1, latent_dim)  # Random latent vector
generated_sample = vae.generate(z_sample)
print("Generated Sample:", generated_sample)

# Encode some input data
mu, log_var = vae.encode(X[:10])  # Encode the first 10 samples
print("Latent Space Mean:", mu)
print("Latent Space Log Variance:", log_var)

# Decode a latent vector
decoded_sample = vae.decode(z_sample)
print("Decoded Sample:", decoded_sample)
