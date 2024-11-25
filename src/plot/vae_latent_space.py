import pandas as pd
import plotly.express as px

# Leer el archivo CSV
df = pd.read_csv("./output/latent_predictions_vae_emojis.csv")

# Definir el array de caracteres
emoji_chars = ['😀', '😄', '😁', '😆', '😅', '🤣', '😂', '🙂', '🙃', '🫠', '😉', '😊', '😇', '🥰', '🤩', '😗', '😚', '🥲', '😋', '😛', '😜', '🤪', '😝', '🤑', '🤗', '🤭', '🫢', '🫣', '🤫', '🤔', '🫡', '🤐', '🤨', '😐', '😑', '😶', '😏', '😒', '🙄', '😬', '😮‍💨', '🤥', '😌', '😔', '😪', '🤤', '😴', '😷', '🤒', '🤕', '🤢', '🤮', '🤧', '🥵', '🥶', '🥴', '😵', '😵‍💫', '🤯', '🤠', '🥳', '🥸', '😎', '🤓', '🧐', '😕', '🫤', '😟', '🙁', '☹️', '☹', '😮', '😯', '😲', '😳', '🥺', '🥹', '😦', '😧', '😨', '😰', '😥', '😢', '😭', '😱', '😖', '😣', '😞', '😓', '😩', '😫', '🥱', '😤', '😡', '😠', '😈', '👿']


# Asumir que las coordenadas en el CSV están en el orden de los caracteres en font3Strings
# Si el CSV tiene más o menos puntos, ajusta el tamaño de font3Strings según sea necesario

# Añadir una nueva columna de 'text' al DataFrame con los caracteres correspondientes
df['text'] = df.index.map(lambda x: emoji_chars[x % len(emoji_chars)])

# Crear el gráfico de dispersión con el texto correspondiente
fig = px.scatter(df, x='x', y='y', title="Gráfico de dispersión de las predicciones latentes",
                 labels={'x': 'X', 'y': 'Y'}, text='text')

# Ajustar la posición del texto para que esté encima del punto
fig.update_traces(textposition='top center')

# Mostrar el gráfico
fig.show()
