import numpy as np
import pandas as pd
import os
import sys
from src import preprocessing
from src import feature_engineering


def master_preprocessing_X(Path_AirportData, Path_WeatherData, Path_Aircraft, Path_correspondance_Aircraft_Airport):

 # Creation of airport and weather dataframe
    X_airport = preprocessing.design_matrix_airport_data(Path_AirportData)
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


def create_target(df):
    df = preprocessing.change_type(df)
    df = pd.DataFrame(df['target'])
    return df


    ## X = final df from master.master_preprocessing
    ## col_numerical = numerical columns of X
    ## col_to_drop = the columns we don't want in X
    ## col_dummies = the colonnes where we want dummys
    ## col_to_target_encode = ['stand','AAC', 'ADG', 'TDG','Wake Category','ATCT Weight Class']
    ## y = y from get_target.get_target
    ## agg_fct is an aggregative fct eg:"mean"
    ## drop is if we don't want the column anymore
def features_pimpage(X,col_numerical,col_to_drop,col_dummies,col_to_target_encode,y,agg_value,path_feature,drop=True,CatBoost=False):

    X = pd.concat([X,y],axis=1)
    X = feature_engineering.master_feature_engineering(X, path_feature)
    X.dropna(subset=['rolling average same runway & same stand'],inplace=True)
    y = pd.DataFrame(X['target'])

    X = feature_engineering.drop_columns(X,col_to_drop)
    X = feature_engineering.imputation(X,col_numerical)
    X = feature_engineering.target_encoding(X,col_to_target_encode,y,agg_value,drop)
    if CatBoost == True:
        X[col_dummies] = X[col_dummies].apply(lambda x: x.astype('str'))
        X[col_numerical] = X[col_numerical].apply(lambda x: x.astype('float'))
        #X[col_to_target_encode] = X[col_to_target_encode].apply(lambda x: x.astype('float'))
    else:
        X = feature_engineering.dummies(X,col_dummies)

    return X,y
