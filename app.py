import numpy as np
import csv
import rtree
import pandas as pd
from haversine import haversine
from model import Model
from with_updates import mlpipeline_update
import math
import importlib
from flask import Flask, request, render_template
from create_lat_lon_in_region import generate_receiver_coordinates


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/tool')
def tool():
    return render_template('tool.html')

@app.route('/generateHeatmap', methods=['POST'])
def generateHeatmap():
    splat = float(request.form.get("splat"))
    splong = float(request.form.get("splong"))
    eplat = float(request.form.get("eplat"))
    eplong = float(request.form.get("eplong"))
    coordinates = [[splat,splong],[eplat,eplong]]
    cleanAISData = open('wittekind', 'r').read()
    exec(cleanAISData)
    generate_receiver_coordinates(min(splat,eplat),min(splong,eplong),max(splat,eplat),max(splong,eplong), 'receiver_c.csv')
    # generate_receiver_coordinates( 10,70.055872 ,10.290600000000001,72.98292479999999 , 'receiver_c.csv')
    twoKm = open('200km.py', 'r').read()
    exec(twoKm)
    predict4 = open('predict-4.py', 'r').read()
    exec(predict4)
    heatmap = open('heatmap.py', 'r').read()
    exec(heatmap)
    threeDScatterMap = open('3d-mapping.py', 'r').read()
    exec(threeDScatterMap)
    threeDPlotMap = open('3d-mapping-Actual.py', 'r').read()
    exec(threeDPlotMap)
    return render_template('outputdisp.html',coordinates=coordinates)


@app.route('/heatmap')
def heatmap():
    return render_template('heatmap.html')

@app.route('/3d_scatter_map')
def scatterMap():
    return render_template('3d_scatter_map.html')

@app.route('/3d_scatter_plot')
def scatterPlot():
    return render_template('3d_scatter_plot.html')


if __name__ == "__main__":
    app.run(debug=True)