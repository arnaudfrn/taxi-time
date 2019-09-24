import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from src.preprocessing import cleaning_airport_df
from itertools import product

# -------------------- Mathieu----------------------------------

def imputation(X,col_numerical):
    #Imputation of all the numerical columns
    #X dataframe coming from master_preprocessing_X
    #col_numerical : list of column numerical
    X.replace({'tbd': np.nan}, inplace=True)
    for i in col_numerical:
        X[i].fillna(X[i].mean(),inplace=True)
    return X

def drop_columns(X,col_to_drop):
    #drop all irrelevant columns
    #X dataframe coming from master_preprocessing_X
    #col_numerical : list of column to drop
    X = X.drop(col_to_drop, axis=1)
    return X

def scaling(X,col_numerical):
    #scaling of all the numerical columns
    #X dataframe coming from master_preprocessing_X
    #col_numerical : list of column numerical
    if len(col_numerical)>0:
        scaler = sklearn.preprocessing.MinMaxScaler()
        X[col_numerical] = scaler.fit_transform(X[col_numerical])
    return X

def dummies(X,col_category) :
    #making dummies of all categorical variable
    #X dataframe coming from master_preprocessing_X
    #col_category : list of column categorical
    X = pd.get_dummies(X, columns=col_category,drop_first=True)
    return X

    ## df = final df from master.master_preprocessing
    ## y = y from get_target.get_target
    ## agg_fct is an aggregative fct eg:"mean"
    ## drop is if we don't want the column anymore
    ## column_list =['stand','AAC', 'ADG', 'TDG','Wake Category','ATCT Weight Class']
def target_encoding(df,column_list,y):
    for column in column_list:
        target_enco_acType = pd.concat([df[column],y], axis=1)
        target_enco_acType = target_enco_acType.groupby(df[column]).agg('mean')
        target_enco_acType = target_enco_acType.to_dict()['target']
        df['target_encoding_'+column]=df[column].map(target_enco_acType)
        df.drop(columns=column, inplace=True)
    return df


def train_test_split(X,y,test_size=10000):
    X_train, X_test = X[0:len(X)-test_size], X[len(X)-test_size:len(X)]
    y_train, y_test = y[0:len(y)-test_size], y[len(y)-test_size:len(y)]
    return X_train,X_test,y_train,y_test


def RF_modeling(X_train,X_test,y_train):
    rf = RandomForestClassifier(n_estimators= 400, min_samples_split= 3, min_samples_leaf= 2, max_features= 'auto',max_depth= 50,bootstrap=True ,random_state=10)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    return pred


def RMSE(y_test,pred):
    pred_in_min = pred/60
    y_test_in_min = [x/60 for x in y_test[0].tolist()]
    calculated_RMSE = np.sqrt(mean_squared_error(pred_in_min,y_test_in_min))
    return RMSE
    ## df = final df from master.master_preprocessing
    ## y = y from get_target.get_target
    ## agg_fct is an aggregative fct eg:"mean"
    ## drop is if we don't want the column anymore

    ## column_list =['stand','AAC', 'ADG', 'TDG','Wake Category','ATCT Weight Class']


# -------------------- Tristan ----------------------------------

## function to get the rolling average of the last 10 planes that landed on the same runway
### input: dataframe (clean one, just before running a model) with a column 'target'
### output: same dataframe with a new column corresponding to this feature
def create_rolling_average_same_runway(df):
    df['runway']=df['runway'].astype(str)
    ra_rw = df[df['runway']==df['runway'].unique()[0]][['aldt','target']].sort_values('aldt').rename(columns={'aldt':'ds','target':'y'}).y.rolling(window=10).mean()
    for rw in df['runway'].unique()[1:]:
        ra_rw = pd.concat([ra_rw, df[df['runway']==rw][['aldt', 'target']].sort_values('aldt').rename(columns={'aldt':'ds','target':'y'}).y.rolling(window=10).mean()])
    ra_rw.name = 'rolling average same runway'
    res = df.join(ra_rw)
    return res

## function to get the rolling average of the last 10 planes that arrived at the same stand
### input: dataframe (clean one, just before running a model) with a column 'target'
### output: same dataframe with a new column corresponding to this feature
def create_rolling_average_same_stand(df):
    df['stand']=df['stand'].astype(str)
    ra_st = df[df['stand']==df['stand'].unique()[0]][['aldt','target']].sort_values('aldt').rename(columns={'aldt':'ds','target':'y'}).y.rolling(window=10).mean()
    for st in df['stand'].unique()[1:]:
        ra_st= pd.concat([ra_st, df[df['stand']==st][['aldt', 'target']].sort_values('aldt').rename(columns={'aldt':'ds','target':'y'}).y.rolling(window=10).mean()])
    ra_st.name = 'rolling average same stand'
    res = df.join(ra_st)
    return res


## function to get the rolling average of the last 5 planes that landed on the same runway and arrived at the same stand
### input: dataframe (clean one, just before running a model) with a column 'target'
### output: same dataframe with a new column corresponding to this feature
def create_rolling_average_same_runway_and_stand(df):
    df['stand']=df['stand'].astype(str)
    df['runway']=df['runway'].astype(str)
    ra_rwst = df[(df['stand']==df['stand'].unique()[0]) &
             (df['runway']==df['runway'].unique()[0])][['aldt','target']].sort_values('aldt').rename(columns={'aldt':'ds','target':'y'}).y.rolling(window=5).mean()
    for rw in df['runway'].unique()[1:]:
        for st in df['stand'].unique()[1:]:
            ra_rwst = pd.concat([ra_rwst, df[(df['stand']==st) &
                                         (df['runway']==rw)][['aldt', 'target']].sort_values('aldt').rename(columns={'aldt':'ds', 'target':'y'}).y.rolling(window=5).mean()])
    ra_rwst.name = 'rolling average same runway & same stand'
    res = df.join(ra_rwst)
    return res



#### MASTER FEATURE ENGINEERING ###
# function that returns the dataframe with the target variable ready for models
## input : ouput of master preprocessing function (dataframe) merged with target (as a column called 'target') and pickle path of the FEATURE THAT TAKES A FUCKING WHILE TO CREATE
## output: dataframe with new variable ready to use
def master_feature_engineering(df, pickle_path):
    df['nb of planes in movement in the plane'] = pd.read_pickle(pickle_path)
    df_augmented = create_rolling_average_same_runway(df)
    df_augmented = create_rolling_average_same_stand(df_augmented)
    df_augmented = create_rolling_average_same_runway_and_stand(df_augmented)
    return df_augmented



def number_of_plane_at_the_gate(df):
    """
    function computing the number of plane at the gate in the closest hour before the flight
    input df: raw df outputed from function clean_airpot_df
    output df: same as input with extra column 'number_of_plane_at_the_gate'
    """
    #Create column time taking both aodt, aibt
    df.loc[:,'time'] = df['aibt']
    df.loc[df['time'].isna(), 'time'] = df.loc[df['time'].isna(), 'aobt']
    df = df[(~df['time'].isna()) & (df['time'].isin(['aibt', 'cibt', 'aldt', 'aobt', 'eldt']))]
    
    #set time to index to groupby
    df.loc[:,'time'] = pd.to_datetime(df['time'])
    df.loc[:,'number_of_plane_at_the_gate'] = 1
    df = df.set_index('time')
    
    #group by 
    grouper = df.groupby([pd.Grouper(freq='1H'), 'stand']).count()
    
    #Get elem of time and stand to create the df on which to merge
    list_date = df.resample('60T').sum().reset_index()['time'].tolist()
    list_stand = df['stand'].reset_index(drop=True).unique()
    
    #create empty df
    comb_df = pd.DataFrame(list(product(list_date, list_stand)), columns=['time', 'stand'])
    
    #merge on empty df and prepare indices
    last_df = (comb_df.merge(grouper.reset_index(), on = ['time', 'stand'], how = 'left')
               .fillna(0)
               .set_index('time')
               .sort_index())
    df = df.drop('number_of_plane_at_the_gate', axis = 1).sort_index()
    df = pd.merge_asof(df.sort_index(), last_df[['stand', 'number_of_plane_at_the_gate']], on = 'time', direction =  'nearest' , by='stand')

    return df

def plane_landing_last_hour(X, y):
    """
    function that takes X dataframe WITH AIBT variable and compute the number of plane that take of in the closest round hour
    return dataframe with new variable number_plane_landing


    #### TO SOLVE #####
    """

    featuredf = pd.concat([y, X], axis = 1)
    featuredf['number_plane_landing'] =  1
    featuredf['aibt'] = pd.to_datetime(featuredf['aibt'])
    
    featuredf = (featuredf.set_index('aibt')
                 .resample('60T')
                 .sum()
                 .sort_index())
    
    X['aibt'] = pd.to_datetime(X['aibt'])
    X = X.set_index('aibt').sort_index()
    
    df = pd.merge_asof(Xfeature, featuredf[['number_plane_landing' ]], on = 'aibt', direction = 'nearest')
    return df

def number_plane_takeoff(X, path_airport):  
    """
    compute the number of plane taking off in the closest hour to landing. 
    input X: df of data preprocessed
    input path_airport: path to airport data
    output df of row len(X)
    """
    df = cleaning_airport_df(Path_AirportData)
    df.loc[:,'number_plane_takeoff'] =  1
    df = df[(~df['aobt'].isna())]
    df['aobt'] = pd.to_datetime(df['aobt'])

    df = (df.set_index('aobt')
                 .resample('60T')
                 .sum()
                 .sort_index())
    X['aibt'] = pd.to_datetime(X['aibt'])
    X = X.set_index('aibt').sort_index()

    df = pd.merge_asof(Xfeature, featuredf[['number_plane_takeoff' ]], on = 'aibt', direction = 'nearest')
    return df

