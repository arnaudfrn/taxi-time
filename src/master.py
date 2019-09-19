import numpy as np
import pandas as pd
import os
import sys
import preprocessing

def master(Path_AirportData, Path_WeatherData, Path_Aircraft):

## Creation of dataframes
    X_airport = preprocessing.design_matrix_airport_data(Path_AirportData)
    X_weather = preprocessing.weather_clean(Path_WeatherData)
    X_merged = preprocessing.join_weather_airport(X_airport, X_weather)






