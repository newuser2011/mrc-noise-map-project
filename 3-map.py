import cesiumpy
import pandas as pd
from opencage.geocoder import OpenCageGeocode

# Replace 'YOUR_API_KEY_HERE' with your actual API key
geolocator = OpenCageGeocode('f6d62c9fe2b0492c970b0784e518220a')

# Read data from CSV file
data = pd.read_csv('noise.csv', header=None)
data = data.dropna()

# create a cesiumpy Viewer
viewer = cesiumpy.Viewer()

# Add a 3D tile layer to the Viewer
tileset_url = "https://assets.cesium.com/17016/tileset.json"
tileset = cesiumpy.Cesium3DTileset(url=tileset_url)
viewer.scene.primitives.add(tileset)

# Add points to the Viewer
for index, row in data.iterrows():
    # Geocode the location of each point
    query = f"{row[0]}, {row[1]}"
    results = geolocator.geocode(query)
    location = results[0]
    description = f"Location: {location['formatted']}, Noise Level: {row[2]}"
    point = cesiumpy.Entity(position=cesiumpy.Cartesian3.from_degrees(row[0], row[1], row[2]), description=description)
    viewer.entities.add(point)

# Set the Viewer camera to a particular location
viewer.camera.flyTo({"destination": cesiumpy.Cartesian3.from_degrees(data[0].mean(), data[1].mean(), 2500.0), "duration": 0})

# Set the Viewer's sky to black
viewer.scene.sky = cesiumpy.Color(0, 0, 0, 1)

# Save the Viewer as an HTML file
viewer.write_html('3d_map.html')
