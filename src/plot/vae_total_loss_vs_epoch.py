import pandas as pd
import plotly.graph_objects as go

# Load the CSV file into a pandas DataFrame
csv_file_start = "./output/vae_"  # Replace with the path to your CSV file


# Create a line plot for epoch vs average_loss using Plotly
fig = go.Figure()
archis = ["400_50_30", "400_150_50_15", "400_200", "400_200_100", "400_400_200_100"]
layers = ["400,50,30","400,150,50,15","400,200","400,200,100","400,400,200,100"]
colors = ["blue", "red", "green", "orange","yellow"]
for archi in reversed(archis):
    data = pd.read_csv(csv_file_start + archi + ".csv")
    fig.add_trace(go.Scatter(
        x=data['epoch'],
        y=data['total_loss'],
        mode='lines',
        name=layers[-1],
        line=dict(color=colors[-1]),
    ))
    layers.pop()
    colors.pop()

# Customize the layout for better readability
fig.update_layout(
    title="Epoch vs Loss",
    xaxis_title="Epoch",
    yaxis_title="Loss",
    font=dict(size=16),  # Set font size
    template="plotly_white",
    width=1000,  # Optional: Set the width of the figure
    height=600,   # Optional: Set the height of the figure
    yaxis=dict(type="log")
)

# Show the plot
fig.show()
