import pandas as pd
import folium
from folium.plugins import HeatMap

# Read data from CSV file
data = pd.read_csv('noise.csv', header=None)
data=data.dropna()
# Convert data to a list of lists
heat_data = [[row[0], row[1], row[2]] for index, row in data.iterrows()]

# Create map centered on the mean coordinates
map_center = [data[0].mean(), data[1].mean()]
heat_map = folium.Map(location=map_center, zoom_start=10)

# Add heat map layer to map
heatmap_layer = HeatMap(heat_data, min_opacity=0.1, radius=15, blur=5)
heat_map.add_child(heatmap_layer)

# Display map
heat_map
heat_map.save('heatmap.html')
heat_map.save('templates/heatmap.html')


def sum_list(lst, start=0, end=None):
    if end is None:
        end = len(lst) - 1
    if start == end:
        return lst[start]
    else:
        mid = (start + end) // 2
        left_sum = sum_list(lst, start, mid)
        right_sum = sum_list(lst, mid + 1, end)
        return left_sum + right_sum
