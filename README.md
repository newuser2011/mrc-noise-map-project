# ML_TL


Implementation of a Neural Network for predicting the Transmission loss between a sender and a reciever both at a depth of 15m from the surface  
* Implemented in numpy, no added dependencies

=====================================  
Description of the files present -   
* speeds folder - speeds data calculated using Leroy  
* Etopo1v2 - bathymetry data  
* area - max depths at each of lat long calculated  
* data.csv - has the rx tx lat long, and tl from both the PE RAM and the NN  
* a.py - extracting the tx rx lat long pairs and store to data.csv  
* extractbathy.py - function to obtain bathymetry data from area.xlsx in the format required for PE RAM model  
* extractspeeds.py - function to obtain speed data from the xlsx in the speeds folder in the format required for PE RAM model  
* runram.py - runs the pyram model, and saves the outputs in data.csv  
* mlpipeline.py - extract data using extractbathy.py and extractspeeds.py and return a 1d array to feed into the NN  
* model.py - contains a class Model, the NN architecture used  
* training.py - train the model from model.py  
* check - error analysis  
* weights folder - has the weights for a 5 layered neural net (hidden layer sizes - 128, 128, 64, 64, 16)
* predict.py - (Untested) script to predict the Transmission Loss between a sender and a reciever
"# test-version" 
