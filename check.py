import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

print("Mean Squared Error -> ", np.mean((df["TL"] - df["NN TL"]) ** 2))
print("Mean Absolute Error -> ", np.mean(np.abs(df["TL"] - df["NN TL"])))

plt.scatter(np.abs(df["TL"] - df["NN TL2"]), df["Distance"])
plt.show()