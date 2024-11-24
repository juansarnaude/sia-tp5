import numpy as np
from abc import ABC, abstractmethod

from src.models.Neuron import Neuron
from src.models.ActivationFunction import ActivationFunction, Tanh
from src.models.Optimizer import Adam

def load_emoji_data():
    # Definimos algunos emojis básicos en formato 7x5
    # 1 representa pixel negro, 0 representa pixel blanco
    
    # Emoji sonriente simple
    smile = [
        [0,1,1,1,0],
        [1,0,0,0,1],
        [1,0,0,0,1],
        [1,0,0,0,1],
        [1,1,0,1,1],
        [1,0,1,0,1],
        [0,1,1,1,0]
    ]
    
    # Emoji triste
    sad = [
        [0,1,1,1,0],
        [1,0,0,0,1],
        [1,0,0,0,1],
        [1,0,0,0,1],
        [1,0,1,0,1],
        [1,1,0,1,1],
        [0,1,1,1,0]
    ]
    
    # Emoji sorprendido
    surprised = [
        [0,1,1,1,0],
        [1,0,0,0,1],
        [1,0,1,0,1],
        [1,0,1,0,1],
        [1,0,1,0,1],
        [1,0,0,0,1],
        [0,1,1,1,0]
    ]
    
    # Emoji guiñando
    wink = [
        [0,1,1,1,0],
        [1,0,0,0,1],
        [1,1,0,0,1],
        [1,0,0,0,1],
        [1,1,0,1,1],
        [1,0,1,0,1],
        [0,1,1,1,0]
    ]
    
    # Convertir cada emoji a un array aplanado
    emojis = [smile, sad, surprised, wink]
    flattened_emojis = []
    
    for emoji in emojis:
        # Convertir la matriz 7x5 en un vector de 35 elementos
        flattened = np.array(emoji).flatten()
        flattened_emojis.append(flattened)
    
    # Convertir a array de numpy
    return np.array(flattened_emojis)

# Función auxiliar para visualizar un emoji (útil para debugging)
def display_emoji(flattened_emoji):
    # Convertir el vector de 35 elementos de vuelta a matriz 7x5
    emoji_matrix = np.reshape(flattened_emoji, (7, 5))
    
    # Imprimir el emoji usando caracteres ASCII
    for row in emoji_matrix:
        for pixel in row:
            print('█' if pixel == 1 else ' ', end='')
        print()


class VAELayer:
    def __init__(self, input_size, output_size, activation_function):
        self.neurons = [Neuron(input_size) for _ in range(output_size)]
        self.activation_function = activation_function
        self.inputs = None
        self.outputs = None
        
    def forward(self, inputs):
        self.inputs = inputs
        neuron_outputs = np.array([neuron.forward(inputs) for neuron in self.neurons])
        self.outputs = self.activation_function(neuron_outputs)
        return self.outputs
    
    def get_weights(self):
        return np.array([neuron.weights for neuron in self.neurons])
    
    def get_biases(self):
        return np.array([neuron.bias for neuron in self.neurons])
    
    def update_weights(self, new_weights):
        for neuron, weights in zip(self.neurons, new_weights):
            neuron.weights = weights
    
    def update_biases(self, new_biases):
        for neuron, bias in zip(self.neurons, new_biases):
            neuron.bias = bias

class VariationalAutoencoder:
    def __init__(self, layer_sizes, activation_function: ActivationFunction, optimizer):
        self.activation_function = activation_function
        self.optimizer = optimizer
        self.encoder_layers = []
        self.decoder_layers = []
        self.latent_dim = layer_sizes[-1]  # Dimensión del espacio latente
        
        # Encoder layers (hasta la capa latente)
        for i in range(len(layer_sizes) - 2):
            self.encoder_layers.append(VAELayer(layer_sizes[i], layer_sizes[i + 1], self.activation_function))

        print(f"size of encoder layers is {len(self.encoder_layers)}")
        
        # Capas para mu y log_var
        self.mu_layer = VAELayer(layer_sizes[-2], self.latent_dim, lambda x: x)  # Sin activación
        self.logvar_layer = VAELayer(layer_sizes[-2], self.latent_dim, lambda x: x)  # Sin activación
        
        # Decoder layers (desde la capa latente)
        decoder_sizes = layer_sizes[-2:][::-1] + layer_sizes[:1]  # Invertimos y agregamos la capa de salida
        print(f"size of decoder layers is {len(decoder_sizes)}")
        print(f"decoder structure: {decoder_sizes}")
        for i in range(len(decoder_sizes) - 1):
            self.decoder_layers.append(VAELayer(decoder_sizes[i], decoder_sizes[i + 1], self.activation_function))
    
    def encode(self, x):
        current_output = x
        for layer in self.encoder_layers:
            current_output = layer.forward(current_output)
        
        mu = self.mu_layer.forward(current_output)
        logvar = self.logvar_layer.forward(current_output)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = np.exp(0.5 * logvar)
        eps = np.random.normal(0, 1, size=mu.shape)
        return mu + eps * std
    
    def decode(self, z):
        current_output = z
        for layer in self.decoder_layers:
            current_output = layer.forward(current_output)
        return current_output
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar):
        # Reconstruction loss (BCE)
        BCE = np.mean((recon_x - x) ** 2)
        
        # KL divergence
        KLD = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar))
        
        return BCE + KLD
    
    def train(self, X, epochs=1000, batch_size=32):
        with open(f"./output/vae_loss_per_epoch.csv", "w") as file:
            file.write("epoch,average_loss\n")
            
            for epoch in range(epochs):
                total_loss = 0
                
                indices = np.arange(len(X))
                np.random.shuffle(indices)
                X_shuffled = X[indices]
                
                for i in range(0, len(X_shuffled), batch_size):
                    batch_X = X_shuffled[i:i + batch_size]
                    
                    # Forward pass
                    recon_batch, mu, logvar = self.forward(batch_X)
                    
                    # Calculate loss
                    loss = self.loss_function(recon_batch, batch_X, mu, logvar)
                    total_loss += loss
                    
                    # Backward pass y actualización de pesos
                    # (Aquí deberías implementar la retropropagación específica para VAE)
                    
                if epoch % 100 == 0:
                    avg_loss = total_loss / (len(X) / batch_size)
                    print(f"Epoch {epoch}, Average Loss: {avg_loss}")
                    file.write(f"{epoch},{avg_loss}\n")
    
    def generate(self, n_samples=1):
        # Generar muestras aleatorias del espacio latente
        z = np.random.normal(0, 1, size=(n_samples, self.latent_dim))
        # Decodificar las muestras
        return self.decode(z)

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar datos de emojis
    X = load_emoji_data()

    #TODO: remove this lines
    #print(f"El tamaño del emoji aplastado es {len(X[0])}")    
    
    # Configurar y entrenar el VAE
    activation = Tanh(input_range=(0, 1), output_range=(0, 1))
    optimizer = Adam(learning_rate=0.0005)
    
    # Arquitectura del VAE
    layer_sizes = [35, 20, 2]  # Ajustar según el tamaño de tus emojis
    
    vae = VariationalAutoencoder(layer_sizes, activation, optimizer)

    #TODO: remove this lines
    # print(f"La dimension del espacion latente es {vae.latent_dim}")
    vae.train(X, epochs=10000, batch_size=32)
    
    # Generar nuevas muestras
    new_samples = vae.generate(n_samples=5)
    
    # Guardar las muestras generadas
    with open("./output/generated_samples.csv", "w") as file:
        for sample in new_samples:
            matrix = np.reshape(sample, (-1, 5))  # Ajustar según dimensiones de tus emojis
            for row in matrix:
                file.write(",".join(map(str, row)) + "\n")
            file.write("\n")
