import pandas as pd
import numpy as np

## -------------------- ARNAUD ----------------------------------
def glossary_feature_selection(path_glossary):
    ignore = ['To ignore', 'to ignore',  'Technical characteristic to ignore', 'Aircraft Type (with another regulation -not to be used for the case)']
    columns_df = pd.read_excel(path_glossary, sheet_name = 1).iloc[:,[0,2]]
    feature_list = [columns_df.iloc[:,0].tolist()[i] for i in range((len(columns_df.iloc[:,1].tolist()))) if columns_df.iloc[:,1].tolist()[i] not in ignore ]
    return feature_list


def create_target(df_airportData):
    df_filter = df_airportData[(~df_airportData['aibt'].isna()) & (~df_airportData['aldt'].isna()) & (df_airportData['aibt'] != 'aibt')].dropna(how = 'all')
    return (pd.to_datetime(df_filter['aibt']) - pd.to_datetime(df_filter['aldt'])).astype('timedelta64[s]')

## ----------------- Mathieu ---------------------------------
def reading_data(PATH,sheet=None):
    # Read csv or xlsx
    if PATH.endswith('xlsx'):
        x = pd.read_excel(PATH,sheet_name=sheet)
    if PATH.endswith('csv'):
        x = pd.read_csv(PATH)
    return x

def filtering_AC_charac(path_correspondance,path_AC_charac):
    # Filtering AC_charac dataset with the aircraft that are actually used in airportdata
    # path_correspondance : path of the pickle of correspondance
    #path_AC_charac : path of the AC charac datasert

    df_charac = reading_data(path_AC_charac,'test')

    matching = pd.read_pickle(path_correspondance)
    matching_list = list(matching.Accharac.values)

    df_charac['Model'] = df_charac['Model'].str.lower().str.strip()
    matching_list = [x.lower().strip() for x in matching_list]
    df_charac = df_charac[df_charac['Model'].isin(matching_list)]
    df_charac.drop_duplicates(subset='Model', keep="first",inplace=True)

    return df_charac

def design_matrix_airport_data(PATH_airport_data):
    #return clean dataset of airport data with index matching the one of the target variable
    df = pd.read_csv(PATH_airport_data)
    df_target = create_target(df)
    df= df.iloc[pd.DataFrame(df_target).index]
    df.drop(columns=['aibt','aldt'],inplace=True)
    return df

## ----------------- Miny ---------------------------------

def weather_clean(path_weather):
    df = pd.read_csv(path_weather, parse_dates=[0])
    df.drop(columns=['PGTM'], inplace=True)
    df = df.groupby('DATE').agg({'AWND': 'mean', 'PRCP': 'sum',
                       'SNOW': 'max','SNWD': 'sum','TAVG': 'mean','TMAX': 'max',
                        'TMIN': 'min','WDF2': 'max','WDF5': 'max',
                        'WSF2': 'max','WSF5': 'max','WT01': 'max',
                        'WT02': 'max','WT03': 'max','WT08':'max'})
    df['SnowProxi'] = (df['PRCP'] > 0.0) & (df['TAVG'] < 45.0)
    df.drop(columns=['SNOW','SNWD'], inplace=True)
    df.fillna(0, inplace=True)
    return df
