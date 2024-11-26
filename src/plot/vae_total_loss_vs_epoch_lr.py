import pandas as pd
import plotly.graph_objects as go

# Load the CSV file into a pandas DataFrame
csv_file_start = "./output/vae_400_200_lr_0_"  # Replace with the path to your CSV file


# Create a line plot for epoch vs average_loss using Plotly
fig = go.Figure()
archis = ["1", "01", "001", "0001"]
lr = ["0.1","0.01","0.001","0.0001"]
colors = ["blue", "red", "green", "orange"]
for archi in reversed(archis):
    data = pd.read_csv(csv_file_start + archi + ".csv")
    fig.add_trace(go.Scatter(
        x=data['epoch'],
        y=data['total_loss'],
        mode='lines',
        name=lr[-1],
        line=dict(color=colors[-1]),
    ))
    lr.pop()
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
