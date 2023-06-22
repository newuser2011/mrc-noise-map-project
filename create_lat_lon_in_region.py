'''import csv

# Define the minimum and maximum latitude and longitude values
arabian_sea_min_lat, arabian_sea_min_lon = 8.122, 66.578
arabian_sea_max_lat, arabian_sea_max_lon = 24.863, 77.485

bay_of_bengal_min_lat, bay_of_bengal_min_lon = 8.122, 77.485
bay_of_bengal_max_lat, bay_of_bengal_max_lon = 21.716, 88.955

# Create a new CSV file and add column headers
with open('reciever-coordinates.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['LAT', 'LON'])

    # Iterate through all possible coordinate pairs with a 0.25 degree spacing
    for lat in range(int(arabian_sea_min_lat*4), int(arabian_sea_max_lat*4)+1):
        for lon in range(int(arabian_sea_min_lon*4), int(arabian_sea_max_lon*4)+1):
            lat_coord = lat/4
            lon_coord = lon/4
            writer.writerow([lat_coord, lon_coord])
    
    for lat in range(int(bay_of_bengal_min_lat*4), int(bay_of_bengal_max_lat*4)+1):
        for lon in range(int(bay_of_bengal_min_lon*4), int(bay_of_bengal_max_lon*4)+1):
            lat_coord = lat/4
            lon_coord = lon/4
            writer.writerow([lat_coord, lon_coord])'''


import csv

def generate_receiver_coordinates(min_lat, min_lon, max_lat, max_lon, file_name):
    # Create a new CSV file and add column headers
    with open(file_name, 'r+') as csvfile:
        csvfile.truncate(0)
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['LAT', 'LON'])

        # Iterate through all possible coordinate pairs with a 0.25 degree spacing
        for lat in range(int(min_lat*4), int(max_lat*4)+1):
            for lon in range(int(min_lon*4), int(max_lon*4)+1):
                lat_coord = lat/4
                lon_coord = lon/4
                writer.writerow([lat_coord, lon_coord])
    print('create_lat_lon_in_region executed')
    
    # Open the file again to delete previous data
    

generate_receiver_coordinates( 10,70.055872 ,10.290600000000001,72.98292479999999 , 'receiver_c.csv')

