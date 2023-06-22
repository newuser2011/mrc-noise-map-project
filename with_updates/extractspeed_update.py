import numpy as np
import pandas as pd
#import matplotib.pyplot as plt

from geographiclib.geodesic import Geodesic
import math

geod = Geodesic.WGS84 #The world ellipsoid

from haversine import haversine

#df = pd.read_csv("data2.csv")
speeds = pd.read_excel("speed/done2007-17_jan_mar_output.xlsx")

def nearest(array1, array2, value1, value2):
    """
    find the nearest lat long present in the speeds df for both the tx and rx
    """
    array1 = np.asarray(array1)
    array2 = np.asarray(array2)
    idx = np.argmin(np.abs(array1 - value1) + np.abs(array2 - value2))

    return idx



def depths():
    """
    Returns Depths for z_ss
    """
    return np.arange(0, 5600, 500)

def extract_speeds(tx_lat, tx_long, rx_lat, rx_long):
    """
    extract the rows pertaining to the tx and rx
    for cw
    """
    f = depths()
    bearing = geod.Inverse(tx_lat, tx_long, rx_lat, rx_long)["azi1"]
    dist = haversine((tx_lat, tx_long), (rx_lat, rx_long))
    update_distances = [0, dist/4, dist/2, 3*dist/4, dist]
    #print(update_distances)
    a = np.zeros((f.shape[0], 5))

    for i in range(len(update_distances)):
        g = geod.Direct(tx_lat, tx_long, bearing, 1000 * update_distances[i])
        lat1 = g["lat2"]
        lon1 = g["lon2"]
        m = nearest(speeds["LATITUDE"], speeds["LONGITUDE"], lat1, lon1)
        #print(m)
        update_distances[i] = haversine((tx_lat, tx_long), (speeds["LATITUDE"][m], speeds["LONGITUDE"][m]))
        for j in range(f.shape[0]):
            a[j, i] = speeds.iloc[m][f[j]]

    return a, update_distances

## REVIEW: Works!!!
#print(extract_speeds(df["Tx lat"][0], df["Tx long"][0], df["Rx lat"][0], df["Rx long"][0]))
