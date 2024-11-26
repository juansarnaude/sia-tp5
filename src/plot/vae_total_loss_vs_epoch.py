import pandas as pd
import plotly.graph_objects as go

# Load the CSV file into a pandas DataFrame
csv_file = "./output/vae_total_loss_through_epochs.csv"  # Replace with the path to your CSV file
data = pd.read_csv(csv_file)

# Create a line plot for epoch vs average_loss using Plotly
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data['epoch'],
    y=data['total_loss'],
    mode='lines',
    name='MSE',
    line=dict(color='blue'),
))

# Customize the layout for better readability
fig.update_layout(
    title="Epoch vs MSE",
    xaxis_title="Epoch",
    yaxis_title="Mean Squared Error (MSE)",
    font=dict(size=16),  # Set font size
    template="plotly_white",
    width=1000,  # Optional: Set the width of the figure
    height=600   # Optional: Set the height of the figure
)

# Show the plot
fig.show()
