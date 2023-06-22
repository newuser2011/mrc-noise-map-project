import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Read data from CSV file
data = pd.read_csv('noise.csv', header=None)
data = data.dropna()

# Create a 3D scatter plot on a world map
fig = go.Figure(data=go.Scattergeo(
    lon=data[1],  # Longitude
    lat=data[0],  # Latitude
    text=data[2],  # Noise Value
    mode='markers',
    marker=dict(
        size=5,
        color=data[2],
        colorscale='Rainbow',
        opacity=0.8,
        colorbar=dict(
            title='Noise Value'
        )
    )
))

# Set the layout of the plot
fig.update_layout(
    title='3D Noise Data Visualization on World Map',
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection_type='equirectangular'
    ),
    scene=dict(
        xaxis=dict(title='Longitude'),
        yaxis=dict(title='Latitude'),
        zaxis=dict(title='Distance'),
        aspectmode='auto'
    )
)

# Save the plot as an HTML file
pio.write_html(fig, file='templates/3d_scatter_map.html', auto_open=True)
print('3d-Scatter-Map executed')
