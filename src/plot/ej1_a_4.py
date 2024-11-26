import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el archivo CSV
data = pd.read_csv('./output/new_characters_autoencoder.csv', header=None)

# Configuración de los caracteres y heatmaps
n_chars = len(data) // 7  # Número total de caracteres (7 filas por carácter)
n_cols = 15  # Número de columnas de heatmaps en la matriz
data_reshaped = data.values.reshape((n_chars, 7, 5))  # Reformatear para 7x5 matrices

# Crear la figura
n_rows = (n_chars + n_cols - 1) // n_cols  # Calcular filas necesarias
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15), gridspec_kw={'wspace': 0.1, 'hspace': 0.3})

# Personalizar la barra de temperatura
cmap = plt.cm.binary  # Tema en blanco y negro
min_val, max_val = 0, 1

# Dibujar cada heatmap en la matriz
for idx, ax in enumerate(axes.flat):
    if idx < n_chars:
        sns.heatmap(data_reshaped[idx], ax=ax, cmap=cmap, cbar=False, vmin=min_val, vmax=max_val)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.axis('off')

# Ajustar una única barra de color para toda la figura
cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # Ajustar posición
norm = plt.Normalize(vmin=min_val, vmax=max_val)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, cax=cbar_ax)

plt.show()
