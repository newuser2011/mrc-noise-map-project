import numpy as np
import pandas as pd
from haversine import haversine

from with_updates.extractspeed_update import depths, extract_speeds
from with_updates.extractbathy_update import extract_bathy


def last_el(array):
    for i in range(len(array)):
        if array[i] == 0:
            break
        a = array[i]
    return a

df = pd.read_csv("data.csv")
#bathy = pd.read_excel("area.xlsx")
#speeds = pd.read_excel("speed/done2007-17_apr_jun_output.xlsx")

#@numba.jit()
def data_pipeline(tx_lat, tx_long, rx_lat, rx_long):
    #returns a 1d numpy array having the data corresponding to the tx, rx pair

    #extract the speed profile
    cw, update_dists = extract_speeds(tx_lat, tx_long, rx_lat, rx_long)
    update_dists = np.array(update_dists)
    for j in range(cw.shape[1]):
        cw[:, j][np.where(cw[:, j]==0)[0]] = last_el(cw[:, j])

    #extract the bathymetry data
    rbzb = extract_bathy(tx_lat, tx_long, rx_lat, rx_long, update_dists)


    #at = attenuation, source: hamilton 1976
    if rbzb[0, 0] < 700:
        at = 0.15
    elif rbzb[0,0] < 1600:
        at = 0.047
    else:
        at = 0.016

    dist = haversine((tx_lat, tx_long), (rx_lat, rx_long))
    dist = dist
    rbzb = rbzb

    #the other vars
    freq = 50
    zr = 15
    zs = 15
    z_sb=np.array([0])
    cb=np.array([1700])
    rhob=np.array([1.5])

    a = np.array(freq)
    a = np.append(a, [tx_lat, tx_long])
    a = np.append(a, [rx_lat, rx_long])
    a = np.append(a, [zr, zs])

    a = np.append(a, cw.reshape(-1, 1))
    a = np.append(a, update_dists.reshape(-1, 1))
    a = np.append(a, dist)

    a = np.append(a, z_sb)
    a = np.append(a, cb)
    a = np.append(a, rhob)
    a = np.append(a, 1.5)
    a = np.append(a, at)
    a = np.append(a, rbzb[:, 1])
    #a = a.reshape(-1, 1)

    return a

'''d = pd.DataFrame()
for i in range(len(df)):
    f = pd.Series(data_pipeline(df["Tx lat"][i], df["Tx long"][i], df["Rx lat"][i], df["Rx long"][i]))
    d = pd.concat([d,f], ignore_index = True,axis=1)

    if i%1000 ==0:
        print(i, "Done")

d.to_csv("mldata_update.csv", index = False)'''
