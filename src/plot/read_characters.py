import numpy as np
import plotly.graph_objects as go

# Leer el archivo CSV
with open("./output/new_characters.csv", "r") as file:
    lines = file.readlines()

# Convertir las líneas del archivo en una matriz numérica
data = np.array([list(map(int, line.strip().split(","))) for line in lines])

# Separar en matrices de 7x5
matrices = [data[i:i + 7] for i in range(0, len(data), 7)]

# Determinar la cantidad de filas y columnas para la cuadrícula
num_matrices = len(matrices)
columns = 5  # Número de columnas en la cuadrícula
rows = (num_matrices // columns) + (num_matrices % columns != 0)

# Crear una matriz grande con el tamaño exacto
matrix_height = 7  # Exactamente 7 filas por matriz
matrix_width = 5   # Exactamente 5 columnas por matriz
giant_matrix = np.zeros((rows * matrix_height, columns * matrix_width))

# Colocar cada matriz en la cuadrícula
for idx, matrix in enumerate(matrices):
    if idx < num_matrices:
        row = idx // columns
        col = idx % columns
        # Calcular la posición de inicio para esta matriz
        start_row = row * matrix_height
        start_col = col * matrix_width
        # Colocar la matriz en la posición correcta
        giant_matrix[start_row:start_row + 7, start_col:start_col + 5] = matrix

# Crear la figura
fig = go.Figure(
    data=go.Heatmap(
        z=giant_matrix,
        colorscale=[[0, 'white'], [1, 'black']],
        showscale=False
    )
)

# Añadir líneas divisorias
shapes = []

# Líneas horizontales (incluyendo márgenes superior e inferior)
for i in range(rows + 1):  # +1 para incluir la línea superior e inferior
    y_position = i * matrix_height - 0.5
    shapes.append(dict(
        type='line',
        x0=-0.5,
        y0=y_position,
        x1=giant_matrix.shape[1] - 0.5,
        y1=y_position,
        line=dict(color="red", width=1)
    ))

# Líneas verticales (incluyendo márgenes izquierdo y derecho)
for i in range(columns + 1):  # +1 para incluir la línea izquierda y derecha
    x_position = i * matrix_width - 0.5
    shapes.append(dict(
        type='line',
        x0=x_position,
        y0=-0.5,
        x1=x_position,
        y1=giant_matrix.shape[0] - 0.5,
        line=dict(color="red", width=1)
    ))

# Configurar el diseño
fig.update_layout(
    title="Caracteres según la salida",
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[-0.5, giant_matrix.shape[1] - 0.5]
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        autorange="reversed",
        range=[-0.5, giant_matrix.shape[0] - 0.5]
    ),
    plot_bgcolor='white',
    shapes=shapes,
    width=800,  # Ancho fijo para mejor visualización
    height=600  # Alto fijo para mejor visualización
)

fig.show()