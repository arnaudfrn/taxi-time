import numpy as np
import pandas as pd

def impute_acType(df):
    """
    function imputing the acType to missing acType using most often used aircraft on the flight number
    
    input: original pd.dataFrame with missing acType
    output: pd.DataFrame with no missing acType
    """
    missing_aircraft =  df[df['acType'].isna()]['flight'].unique().tolist()
    
    df_flight_impute = (df[df['flight'].isin(missing_aircraft)]
                                        .groupby(['flight'])['acType']
                                        .value_counts()
                                        .rename("nb")
                                        .to_frame()
                                        .reset_index())
    
    df_flight_impute = df_flight_impute.loc[df_flight_impute.groupby(['flight'])["nb"].idxmax()].drop('nb', axis = 1)
    df_final = df[(df['acType'].isna()) & (~df['flight'].isna())].drop('acType', axis = 1).merge(df_flight_impute, how = 'left', on = ["flight"])

    return df_final