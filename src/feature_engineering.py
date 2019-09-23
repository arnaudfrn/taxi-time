import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# -------------------- Mathieu----------------------------------

def imputation(X,col_numerical):
    #Imputation of all the numerical columns
    #X dataframe coming from master_preprocessing_X
    #col_numerical : list of column numerical
    X.replace({'tbd': np.nan}, inplace=True)
    for i in col_numerical:
        X[i].fillna(X[i].median(),inplace=True)
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

def drop_imput_scaled_dummies(X,col_numerical,col_to_drop,col_category):
     X = drop_columns(X,col_to_drop)
     X = imputation(X,col_numerical)
     #X = scaling(X,col_numerical)
     X = dummies(X,col_category)
     return X



def train_test_split(X,y,test_size=1000):
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
    calculated_RMSE = np.sqrt(sk.metrics.mean_squared_error(pred_in_min,y_test_in_min))
    return RMSE

    ## df = final df from master.master_preprocessing
    ## y = y from get_target.get_target
    ## agg_fct is an aggregative fct eg:"mean"
    ## drop is if we don't want the column anymore

    ## column_list =['stand','AAC', 'ADG', 'TDG','Wake Category','ATCT Weight Class']

def traget_encoding(df,column_list,y,agg_fct,drop=False):
    for column in column_list:
        target_enco_acType = pd.concat([df[column],y], axis=1)
        target_enco_acType = target_enco_acType.groupby(df[column]).agg(agg_fct)
        target_enco_acType = target_enco_acType.to_dict()['target']
        df['target_encoding_'+column]=df[column].map(target_enco_acType)
        if drop == True:
            df.drop(columns=column, inplace=True)
        else:
            continue
        return df