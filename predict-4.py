# import libraries
import pandas as pd
import numpy as np
from haversine import haversine
from model import Model
from with_updates import mlpipeline_update
import math
import csv
import timeit
from heatmap import sum_list
#import timeit
# load the pretrained model
model = Model(39, 128, 128, 64, 64, 16)
b1 = np.load("with_updates\weights\\b1.npy")
b2 = np.load("with_updates\weights\\b2.npy")
b3 = np.load("with_updates\weights\\b3.npy")
b4 = np.load("with_updates\weights\\b4.npy")
b5 = np.load("with_updates\weights\\b5.npy")
b6 = np.load("with_updates\weights\\b6.npy")
w1 = np.load("with_updates\weights\w1.npy")
w2 = np.load("with_updates\weights\w2.npy")
w3 = np.load("with_updates\weights\w3.npy")
w4 = np.load("with_updates\weights\w4.npy")
w5 = np.load("with_updates\weights\w5.npy")
w6 = np.load("with_updates\weights\w6.npy")
model.set_weights([w1, b1, w2, b2, w3, b3, w4, b4, w5, b5])

# load the input data from input.csv file
ais_data = pd.read_csv("cleaned_AIS_Data.csv", low_memory=False)
# print(ais_data.head())
input_data = pd.read_csv("receiver_ship.csv",low_memory=False)
# print(input_data.head())
# initialize an empty list to store predicted values
predictions = []
ans=[]

# Read the column names from the file
filename = 'noise.csv'
with open(filename, 'r') as file:
    columns = file.readline()

# Truncate the file to delete the old data
with open(filename, 'w') as file:
    file.write(columns)

print(input_data.iloc[1,0][3])

'''def calc_source_level(row):
    k = 4.1 # constant for open water conditions
    rho = 1025 # density of seawater in kg/m^3
    area = float(row[0])**2 * np.pi/4 # effective area of propeller
    thrust = float(row[1])* 9.81 # thrust in newtons
    power_per_engine = float(row[2]) / float(row[4]) # power per engine in kW
    power_density = power_per_engine / (area * float(row[5])) # power density in kW/m^2
    cavitation_no = float(row[5]) / float(row[6])
    block_coeff = float(row[8])
    dwt = float(row[3])
    froude_no = float(row[7])
    # Check for division by zero or negative values
    if (thrust == 0 or block_coeff <= 0 or cavitation_no <= 0 or cavitation_no >= 1 or dwt <= 0 or froude_no <= 0 or area <= 0):
        return np.nan
    else:
        source_level = 10 * np.log10(k**2 * rho * area * power_density * dwt * froude_no**3 / thrust * (1 - cavitation_no**3) / block_coeff**2)
        # Check for negative values
        if source_level < 0:
            return np.nan
        else:
            return source_level'''




# def search_ais_data_file(lat, lon):
#     # Read in the AIS data file
    
#     # Create a dictionary to map lat, lon pairs to the rows that contain them
#     lat_lon_dict = {}
#     for index, row in ais_data.iterrows():
#         lat_lon = (row[9], row[10])
#         if lat_lon not in lat_lon_dict:
#             lat_lon_dict[lat_lon] = []
#         lat_lon_dict[lat_lon].append(row[15])
    
#     # Search for the given lat, lon pair in the dictionary
#     source_levels = lat_lon_dict.get((lat, lon), [])
#     source_level = float(source_levels[0]) if source_levels else None
#     return source_level



# def sum_list(lst, start=0, end=None):
#     if end is None:
#         end = len(lst) - 1
#     if start == end:
#         return lst[start]
#     else:
#         mid = (start + end) // 2
#         left_sum = sum_list(lst, start, mid)
#         right_sum = sum_list(lst, mid + 1, end)
#         return left_sum + right_sum

# def sum_list_no_nan(lst):
#     lst_no_nan = [x for x in lst if not math.isnan(x)]
#     if len(lst_no_nan) == 0:
#         return 0
#     else:
#         return sum_list(lst_no_nan)





# use data pipeline function to transform input data into X array
for i in range(0,len(input_data.iloc[:,0])):
    running_sum=0
    rx_lat=float(input_data.iloc[i,0][1:-1].split(",")[0])
    rx_long=float(input_data.iloc[i,0][1:-1].split(",")[1])

    for j in range(1,len(input_data.iloc[i,:])):
         if not pd.isna(input_data.iloc[i,j]):
            tx_lat=float(input_data.iloc[i,j][1:-1].split(",")[0])
            tx_long=float(input_data.iloc[i,j][1:-1].split(",")[1])
            #start = timeit.timeit()
            X = mlpipeline_update.data_pipeline(tx_lat, tx_long, rx_lat, rx_long)
            #end = timeit.timeit()
            #print(end)
            #timetaken=timeit.timeit(X,number=10)
            #print(timetaken)
            model.forward(X, p=1)
            y = model.z3
            predictions.append(y)
            average_prediction = np.mean(predictions)

            #print(average_prediction)
            # read the cleaned AIS data file into a pandas dataframe

            # initialize an empty list to store the rows that match the search criteria
            
            
            # loop through each row in the AIS data file
            '''for index, row in ais_data.iterrows():
                # extract the latitude and longitude values from the row
                lati = row[9]
                lon = row[10]
                # check if the latitude and longitude values match the search criteria
                if (lati == tx_lat) and (lon == tx_long):
                    # if a match is found, append the row to the list of matching rows
                    s_float = pd.to_numeric(row, errors='coerce').astype(float)'''
            
            lat_lon_dict = {}
            for index, row in ais_data.iterrows():
                lat_lon = (row[9], row[10])
                if lat_lon not in lat_lon_dict:
                    lat_lon_dict[lat_lon] = []
                lat_lon_dict[lat_lon].append(row[15])
            

            source_levels = lat_lon_dict.get((tx_lat, tx_long), [])
            sl = float(source_levels[0]) if source_levels else None

            # sl=search_ais_data_file(tx_lat,tx_long)
            
            ans.append(abs(sl-average_prediction))
           
    
    
    # result=sum_list_no_nan(ans)
    lst_no_nan = [x for x in ans if not math.isnan(x)]
    if len(lst_no_nan) == 0:
        result = 0
    else:
        result = sum_list(lst_no_nan)

    print(result)
    ans.clear()                
                    
# Manually truncate the file to delete old data
    

    # Create a new DataFrame with the data
    df = pd.DataFrame({'rx_lat': [rx_lat], 'rx_lon': [rx_long], 'result': [result]})

    # Append the new data to the CSV file
    df.to_csv(filename, index=False, mode='a', header=False)


print("predict-4 executed")
    
                  
            


            


            