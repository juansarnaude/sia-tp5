import numpy as np
from src.models.MultiLayerPerceptron import MultiLayerPerceptron

class VariationalAutoencoder:
    def __init__(self, latent_dim, encoder_layers, decoder_layers, activation, optimizer1, optimizer2):
        self.latent_dim = latent_dim
        self.activation = activation

        self.encoder = MultiLayerPerceptron(
            encoder_layers + [2 * latent_dim],
            activation,
            optimizer1,
        )

        self.decoder = MultiLayerPerceptron(
            [latent_dim] + decoder_layers,
            activation,
            optimizer2,
        )

    def reparameterize(self, mean, log_var):
        epsilon = np.random.normal(size=mean.shape)  # Ensure size matches `mean`
        std = np.exp(0.5 * log_var)  # Standard deviation
        z = mean + std * epsilon
        return z, epsilon

    def loss_function(self, x, reconstructed_x, mu, log_var):
        reconstruction_loss = np.mean(np.square(x - reconstructed_x))
        kl_loss = -0.5 * np.sum(1 + log_var - np.square(mu) - np.exp(log_var))
        return reconstruction_loss + kl_loss

    def train(self, X, epochs=100, batch_size=32):
        z_list = []
        total_loss_list = []

        for epoch in range(epochs):
            total_loss = 0
            np.random.shuffle(X)

            for i in range(0, len(X), batch_size):
                batch = X[i:i + batch_size]

                batch_means_list = []
                batch_log_var_list = []
                reconstructed_batch = []

                current_z = []

                for x in batch:
                    encoded = self.encoder.feed_forward(x)

                    mean = encoded[: len(encoded) // 2]
                    log_var = encoded[len(encoded) // 2:]


                    batch_means_list.append(mean)
                    batch_log_var_list.append(log_var)

                    z, epsilon = self.reparameterize(mean, log_var)

                    current_z.append(z)

                    reconstructed = self.decoder.feed_forward(z)
                    reconstructed_batch.append(reconstructed)

                batch_mean = np.mean(batch_means_list)  #TODO THIS MAY BE WRONG
                batch_log_var = np.mean(batch_log_var_list)

                reconstructed_batch = np.vstack(reconstructed_batch)

                reconstruction_loss = np.mean(np.square(batch - reconstructed_batch))
                kl_loss = -0.5 * np.sum(1 + batch_log_var - np.square(batch_mean) - np.exp(batch_log_var))
                batch_loss = reconstruction_loss + kl_loss
                total_loss += batch_loss

                decoder_weight_gradients = [np.zeros_like(layer.get_weights()) for layer in self.decoder.layers]
                decoder_bias_gradients = [np.zeros_like(layer.get_biases()) for layer in self.decoder.layers]

                error_decoder = []

                for inp, exp in zip(current_z, batch):
                    weight_gradients, bias_gradients = self.decoder.backpropagation(inp, exp)

                    reconstructed = self.decoder.feed_forward(inp)
                    error_decoder.append(exp - reconstructed)

                    for layer_idx in range(len(self.decoder.layers)):
                        decoder_weight_gradients[layer_idx] += weight_gradients[layer_idx]
                        decoder_bias_gradients[layer_idx] += bias_gradients[layer_idx]

                batch_size = len(batch)
                decoder_weight_gradients = [grad / batch_size for grad in decoder_weight_gradients]
                decoder_bias_gradients = [grad / batch_size for grad in decoder_bias_gradients]

                decoder_last_delta = np.array(error_decoder) * self.decoder.activation_function(reconstructed_batch,
                                                                                                derivative=True)

                dz_dmean = np.ones_like(batch_means_list)
                dz_dstd = epsilon * np.ones_like(batch_means_list)

                mean_error = np.dot(decoder_last_delta.T, dz_dmean)
                std_error = np.dot(decoder_last_delta.T, dz_dstd)

                encoder_reconstruction_error = np.concatenate((mean_error, std_error), axis=1)

                encoder_reconstruction_gradients_weights, encoder_reconstruction_gradients_biases = self.encoder.backpropagation_vae(encoder_reconstruction_error)

                dL_dm = batch_means_list
                dL_dv = 0.5 * (np.exp(batch_log_var_list) - 1)
                encoder_kl_error = np.concatenate((dL_dm, dL_dv), axis=1)
                encoder_kl_gradients, encoder_kl_biases_gradients = self.encoder.backpropagation_vae(encoder_kl_error)

                encoder_weight_gradients = [
                    g1 + g2
                    for g1, g2 in zip(encoder_kl_gradients, encoder_reconstruction_gradients_weights)
                ]

                encoder_biases_gradients = [
                    g1 + g2
                    for g1, g2 in zip(encoder_kl_biases_gradients, encoder_reconstruction_gradients_biases)
                ]

                self.encoder.update_weights(encoder_weight_gradients, encoder_biases_gradients)
                self.decoder.update_weights(decoder_weight_gradients, decoder_bias_gradients)

                z_list.append(current_z)

            print(f"Epoch: {epoch}, Total Loss: {total_loss:.4f}, Reconstruction Loss: {reconstruction_loss:.4f}, KL Loss: {kl_loss:.4f}")
            total_loss_list.append(total_loss)

        return z_list, total_loss_list

    def generate(self, z_sample):
        return self.decoder.feed_forward(z_sample)

    def encode(self, X):
        encoded = self.encoder.feed_forward(X)

        mu = encoded[: len(encoded) // 2]
        log_var = encoded[len(encoded) // 2:]

        z, _ = self.reparameterize(mu, log_var)
        return z


    def decode(self, z):
        return self.decoder.feed_forward(z)
