import pandas as pd
import plotly.graph_objects as go

# Nombres de los archivos .csv
file_names = [
    "./output/mse_vs_epoch_1.csv",
    "./output/mse_vs_epoch_2.csv",
    "./output/mse_vs_epoch_3.csv",
    "./output/mse_vs_epoch_4.csv",
    "./output/mse_vs_epoch_5.csv"
]

# Crear una figura de Plotly
fig = go.Figure()

# Colores y nombres de referencia para cada archivo
colors = ['red', 'blue', 'green', 'orange', 'purple']
labels = ['[20]', '[20,20]', '[20,20,20]', '[20,20,20,20]', '[20,20,20,20,20]']

# Iterar sobre los archivos y agregar una línea por archivo
for file_name, color, label in zip(file_names, colors, labels):
    # Leer el archivo .csv
    data = pd.read_csv(file_name)
    
    # Agregar la línea al gráfico
    fig.add_trace(go.Scatter(
        x=data['epoch'],
        y=data['average_loss'],
        mode='lines',
        line=dict(color=color),
        name=label
    ))

# Configurar el diseño del gráfico
fig.update_layout(
    title="Comparación de MSE vs Épocas para distinta cantidad de layers",
    xaxis_title="Épocas",
    yaxis_title="MSE (Mean Square Error)",
    legend_title="Datasets",
    template="plotly_white"
)

# Mostrar el gráfico
fig.show()
