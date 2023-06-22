import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Read data from CSV file
data = pd.read_csv('noise.csv', header=None)
data = data.dropna()

# Create a 3D scatter plot
fig = go.Figure(data=go.Scatter3d(
    x=data[1],  # Longitude
    y=data[0],  # Latitude
    z=data[2]/100,  # Noise value
    mode='markers',
    marker=dict(
        size=5,
        color=data[2],
        colorscale='Viridis',
        opacity=0.8
    )
))

# Set the layout of the plot
fig.update_layout(
    title='Noise Data Visualization',
    scene=dict(
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        zaxis_title='Distance'
    )
)

# Display the plot
# fig.show()


# Save the plot as an HTML file
pio.write_html(fig, file='templates/3d_scatter_plot.html', auto_open=True)
print('3d-Plot-Map executed')