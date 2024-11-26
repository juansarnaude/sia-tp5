import pandas as pd
import plotly.express as px

# Leer el archivo CSV
df = pd.read_csv("./output/latent_predictions_autoencoder.csv")

# Definir el array de caracteres
font3Strings = [
    "`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
    "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", "DEL"
]

# Añadir una nueva columna de 'text' al DataFrame con los caracteres correspondientes
df['text'] = df.index.map(lambda x: font3Strings[x % len(font3Strings)])

# Crear el gráfico de dispersión con el texto correspondiente
fig = px.scatter(df, x='x', y='y', title="Gráfico de dispersión de las predicciones en el espacio latente",
                 labels={'x': 'X', 'y': 'Y'}, text='text')

# Ajustar la posición del texto y el tamaño de la letra de los puntos
fig.update_traces(textposition='top center', textfont=dict(size=16))  # Cambia el tamaño con `size`

# Configurar el tamaño del título y los números de los ejes
fig.update_layout(
    title=dict(text="Gráfico de dispersión de las predicciones en el espacio latente", font=dict(size=24)),
    xaxis=dict(title=dict(font=dict(size=18)), tickfont=dict(size=14)),
    yaxis=dict(title=dict(font=dict(size=18)), tickfont=dict(size=14))
)

# Mostrar el gráfico
fig.show()
