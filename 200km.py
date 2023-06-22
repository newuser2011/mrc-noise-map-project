import pandas as pd
import numpy as np
import os
import rtree

# Load the receiver coordinates from "receiver-coordinates.csv"
receiver_df = pd.read_csv("receiver_c.csv")

# Load the AIS data from "cleaned_AIS_Data.csv"
ais_df = pd.read_csv("cleaned_AIS_Data.csv")



# Create an R-tree index for the AIS data
idx = rtree.index.Index()
for i, row in ais_df.iterrows():
    idx.insert(i, (row["LON"], row["LAT"], row["LON"], row["LAT"]))

# Define a function to calculate the haversine distance between two points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    a = np.sin(dLat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dLon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c

# Create an empty dictionary to store the ship coordinates for each receiver
receiver_ships = {}

# Loop through each receiver coordinate and find all ships within 200km
for i, row in receiver_df.iterrows():
    receiver_lat = row["LAT"]
    receiver_lon = row["LON"]
    ship_coords = []
    # Calculate the bounding box for the search
    lon_min, lat_min, lon_max, lat_max = (
        receiver_lon - 2, receiver_lat - 2, receiver_lon + 2, receiver_lat + 2)
    # Use the R-tree index to find all ships within the bounding box
    candidate_ships = list(idx.intersection((lon_min, lat_min, lon_max, lat_max)))
    for j in candidate_ships:
        ship_lat = ais_df.loc[j, "LAT"]
        ship_lon = ais_df.loc[j, "LON"]
        distance = haversine(receiver_lat, receiver_lon, ship_lat, ship_lon)
        if distance < 200:
            ship_coords.append((ship_lat, ship_lon))
    # Store the ship coordinates in the dictionary using the receiver coordinates as the key
    receiver_ships[(receiver_lat, receiver_lon)] = ship_coords

# Convert the dictionary to a DataFrame
df = pd.DataFrame.from_dict(receiver_ships, orient="index",dtype=object)
df = df.reset_index()
df = df.rename(columns={"index": "RECEIVER_COORDINATES"})
num_cols = len(df.columns) - 1
new_col_names = ["SHIP_COORDINATES_" + str(i+1) for i in range(num_cols)]
df = df.rename(columns=dict(zip(df.columns[1:], new_col_names)))

# Write the DataFrame to a CSV file
df.to_csv("receiver_ship.csv", index=False)
print("200km executed")


'''The rtree module is imported.
An R-tree index is created using the idx = rtree.index.Index() command.
 The index is built by looping through the AIS data and inserting each row into the index using 
 idx.insert().
The search for nearby ships is optimized by using the R-tree index. The bounding box for the search 
is calculated using the receiver coordinates and a fixed search radius of 2 degrees. The index is queried 
using idx.intersection() to find all AIS rows that fall within the bounding box. 
The distance is then calculated only for the   An empty dictionary receiver_ships is created to store 
the ship coordinates for each receiver.After finding the nearby ships for each receiver, the ship 
coordinates are stored in the receiver_ships dictionary using the receiver coordinates as the key.
The receiver_ships dictionary is then converted to a pandas DataFrame using the pd.DataFrame.from_dict()
 method. The resulting DataFrame has the receiver coordinates as the first two columns and the ship 
 coordinates in the remaining columns.
Finally, the DataFrame is written to an Excel file using the df.to_excel() method. 
The index=False argument is used to exclude the row indices from the output.   
The df.rename() method now maps a dictionary that has only one key-value pair, where the key is the old 
column name ("index") and the value is the new column name ("RECEIVER_COORDINATES").
The number of new column names is now dynamically generated using the num_cols variable,'''