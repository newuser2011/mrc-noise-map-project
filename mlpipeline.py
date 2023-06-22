import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from haversine import haversine
import numba

from extractspeed import depths, extract_speeds, update_speeds
from extractbathy import extract_bathy


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
    cw = extract_speeds(tx_lat, tx_long, rx_lat, rx_long)
    cw[:, 0][np.where(cw[:, 0]==0)[0]] = last_el(cw[:, 0])
    cw[:, 1][np.where(cw[:, 1]==0)[0]] = last_el(cw[:, 1])
    
    #extract the bathymetry data
    rbzb = extract_bathy(tx_lat, tx_long, rx_lat, rx_long)


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
    d = pd.concat([d, f], ignore_index=True, axis=1)


    if i%1000 ==0:
        print(i, "Done")

d.to_csv("data2.csv", index = False)'''



'''This code appears to be a data processing pipeline that extracts information from various data sources,
 such as speed profiles and bathymetry data, and then combines them with other parameters to create a 1D 
 numpy array. The code then iterates through the rows of a CSV file named "data.csv" and applies this
  pipeline to extract the required information for each row. The extracted information is then stored in 
  a pandas dataframe named "d" and is saved to a CSV file named "mldata.csv".

The pipeline involves the use of the haversine formula to calculate the distance between two sets of 
latitude and longitude coordinates, and the use of the "extract_speeds" and "extract_bathy" functions from 
external modules to extract speed profile and bathymetry data respectively. The pipeline also contains 
some conditional statements that modify some of the extracted data based on certain conditions.

There is a commented out section that suggests that the pipeline might have been designed to read data 
from Excel files rather than CSV files, but this is not currently being used.

Overall, it appears that this code is designed to extract and process data for use in some kind of 
machine learning application, given that the final output is saved to a CSV file that could be used for 
training a machine learning model.



'''