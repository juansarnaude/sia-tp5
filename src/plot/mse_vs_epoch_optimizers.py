import pandas as pd
import plotly.graph_objects as go

# Nombres de los archivos .csv con el prefijo del directorio
file_names = [
    "./output/mse_vs_epoch_adam.csv",
    "./output/mse_vs_epoch_momentum.csv",
    "./output/mse_vs_epoch_gradient_descent.csv"
]

# Referencias y colores para cada optimizador
references = [
    "Adam (beta1=0.9, beta2=0.999, epsilon=1e-8)",
    "Momentum(momentum=0.9)",
    "Gradient Descent"
]
colors = ['blue', 'green', 'red']

# Crear una figura de Plotly
fig = go.Figure()

# Iterar sobre los archivos y agregar una línea por archivo
for file_name, reference, color in zip(file_names, references, colors):
    # Leer el archivo .csv
    data = pd.read_csv(file_name)
    
    # Agregar la línea al gráfico
    fig.add_trace(go.Scatter(
        x=data['epoch'],
        y=data['average_loss'],
        mode='lines',
        line=dict(color=color),
        name=reference
    ))

# Configurar el diseño del gráfico con texto legible
fig.update_layout(
    title="Comparación de MSE vs Épocas para Distintos Optimizadores",
    xaxis_title="Épocas",
    yaxis_title="MSE (Mean Square Error)",
    legend_title="Optimizadores",
    template="plotly_white",
    font=dict(
        size=16  # Tamaño de la fuente para mejor legibilidad
    )
)

# Mostrar el gráfico
fig.show()
