import numpy as np
import pandas as pd
from haversine import haversine

from geographiclib.geodesic import Geodesic
import math

geod = Geodesic.WGS84 #The world ellipsoid

bathy = pd.read_excel("area.xlsx")
#df = pd.read_csv("data.csv")

def nearest(array1, array2, value1, value2):
    """
    find the nearest lat long present in the speeds df for both the tx and rx
    """
    array1 = np.asarray(array1)
    array2 = np.asarray(array2)
    idx = np.argmin(np.abs(array1 - value1) + np.abs(array2 - value2))

    return idx

def extract_bathy(tx_lat, tx_long, rx_lat, rx_long, distances):
    """
    extract the depths pertaining to tx and rx
    for rbzb
    """
    a = np.zeros((len(distances), 2))
    bearing = geod.Inverse(tx_lat, tx_long, rx_lat, rx_long)["azi1"]
    for i in range(len(distances)):
        a[i, 0] = distances[i]
        g = geod.Direct(tx_lat, tx_long, bearing, distances[i])
        lat1 = g["lat2"]
        lon1 = g["lon2"]
        m = nearest(bathy["LATITUDE"], bathy["LONGITUDE"], lat1, lon1)
        a[i, 1] = bathy["Dmax"][m]

    a[:, 1] = np.abs(a[:, 1])

    return a

#Check - Works!!!
#print(extract_bathy(df["Tx lat"][0], df["Tx long"][0], df["Rx lat"][0], df["Rx long"][0], [0, 9058.617193525713, 36311.404473112, 70147.63925898903, 106562.91707912697, 131280.0097300111]))
