import numpy as np
import pandas as pd
import os
import sys
from src import preprocessing


def master_preprocessing_X(Path_AirportData, Path_WeatherData, Path_Aircraft, Path_correspondance_Aircraft_Airport):

 # Creation of airport and weather dataframe
    X_airport = preprocessing.design_matrix_airport_data(Path_AirportData)

    ### X_airport = get_df_obs1(cleaning_airport_df(Path_AirportData)) (same with Tristan's functions)

    X_weather = preprocessing.weather_clean(Path_WeatherData)

# Merging both dataframes
    X_merged = preprocessing.join_weather_airport(X_airport, X_weather)

# Creation of AC characteristics dataframe table of correspondances
    df_charac = preprocessing.filtering_AC_charac(Path_correspondance_Aircraft_Airport, Path_Aircraft)
    matching_dict = preprocessing.correspondance_dictionnary(Path_correspondance_Aircraft_Airport)

# Merging the three dataset received
    X_final = preprocessing.augmented_design_matrix_with_AC_charac(X_merged, df_charac, matching_dict)

# Drop useless columns    
    X_final = X_final.drop(['date'], axis=1)

    return X_final




########## Miny's Test ############
#import os
# import sys
# import pandas as pd
# sys.path.append("..")
# from src import preprocessing
# from src import master
# master.master_preprocessing_X('../../0. Airport data/Airport_Data.csv',
#                               '../../2. Weather data/weather_data_prep.csv',
#                               '../../1. AC characteristics/ACchar.xlsx',
#                               '../Correspondance.pkl')
####################################