import pandas as pd
import numpy as np
from haversine import haversine
from pyram.PyRAM import PyRAM
import matplotlib.pyplot as plt

from extractspeed_update import depths, extract_speeds
from extractbathy_update import extract_bathy

df = pd.read_csv("data2.csv")
bathy = pd.read_excel("area.xlsx")
speeds = pd.read_excel("speed/done2007-17_jan_mar_output.xlsx")

def near(array, value):
    """
    gets the nearest range for which tl is calculated to the actual distance
    """
    array= np.asarray(array)
    idx = np.argmin(np.abs(array - value))

    return idx

def last_el(array):
    """
    Extends the array by substituting the last element
    """
    x = 0
    for i in range(len(array)):
        if array[i] == 0:
            break
        x = array[i]
    return x


def run_pyram(dist, z_ss, rp_ss, cw, attn, rbzb, dr = 500, dz = 500, c0 = 1600):
    """
    function to run the pyram model
    returns the transmission loss between tx and rx
    """
    freq = 50
    zr = 15
    zs = 15
    z_sb=np.array([0])
    rp_sb=np.array([0])
    cb=np.array([[1700]])
    rhob=np.array([[1.5]])
    pyram = PyRAM(freq, zs, zr, z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob, attn, rbzb, dr = dr, dz = dz, c0 = c0)

    pyram.run()

    m = near(pyram.vr, 1000 * dist)

    return pyram.tll[m]

#here we fit it all together
trans_loss = []
for i in range(1000):
    #get the lat long
    tx_lat, tx_long = df["Tx lat"][i], df["Tx long"][i]
    rx_lat, rx_long = df["Rx lat"][i], df["Rx long"][i]
    dist = df["Distance"][i]

    #Get the sound speed column values and the update ranges
    cw, update_dists = extract_speeds(tx_lat, tx_long, rx_lat, rx_long)

    #get rbzb
    rbzb = extract_bathy(tx_lat, tx_long, rx_lat, rx_long, update_dists)


    #get z_ss
    z_ss = depths()


    #get rp_ss
    #rp_ss = update_speeds(tx_lat, tx_long, rx_lat, rx_long)
    rp_ss = np.array(update_dists) * 1000

    #Account for zeros in cw
    for j in range(cw.shape[1]):
        cw[:, j][np.where(cw[:, j]==0)[0]] = last_el(cw[:, j])

    #attn - source: Hamilton 1976
    if rbzb[0, 0] < 700:
        at = 0.15
    elif rbzb[0,0] < 1600:
        at = 0.047
    else:
        at = 0.016
    attn=np.array([[at]])
    tl = run_pyram(dist, z_ss, rp_ss, cw, attn, rbzb)
    trans_loss.append(tl)


    if i%100 == 0:
        print(i, "Done")
#df["TL"] = trans_loss

#df.to_csv("data2.csv", index = False)
plt.scatter(df["Distance"][:1000], trans_loss)
plt.show()
