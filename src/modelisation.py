import pandas as pd 
import numpy as np

def compute_rmse(y, y_pred):
    """
    function computing RMSE 

    input: real target values and predicted values
    output : root mean squared error in minutes
    """
    return np.sqrt(((y_pred - y)**2).mean()) / 60

def compute_mape(y, y_pred):
    """
    function computing MAPE 

    input: real target values and predicted values
    output : root mean squared error in minutes
    """
    return np.mean(np.abs((y - y_pred) / y)) * 100 / 60

def compute_mae(y, y_pred):
    """
    function computing MAE 

    input: real target values and predicted values
    output : root mean squared error in minutes
    """
    return np.mean(np.abs((y - y_pred)))/60

def tenth_percentile(y_test, y_pred) :
    """
    Compute the value of the 10th worse error 

    input: real target values and predicted values
    output : value of the 10th percentile error
    """
    return np.abs(y_test - y_pred).sort_values().iloc[int(len(y_test)*0.10)]

def ninetieth_percentile(y_test, y_pred) :
    """
    Compute the value of the 90th worse error 

    input: real target values and predicted values
    output : value of the 90th percentile error
    """
    return np.abs(y_test - y_pred).sort_values().iloc[int(len(y_test)*0.90)]

def compute_metrics(y_test, y_pred) : 
    """
    Create a df with all the metrics for a specific result 

    input: real target values and predicted values
    output : df with the different metrics for a specific y_test / y_pred couple
    """
    metrics = pd.DataFrame()
    metrics['RMSE'] = [compute_rmse(y_test, y_pred)]
    metrics['MAPE'] = [compute_mape(y_test, y_pred)]
    metrics['MAE'] = [compute_mae(y_test, y_pred)]
    metrics['tenth_perc'] = [tenth_percentile(y_test, y_pred)]
    metrics['ninetieth_perc'] = [ninetieth_percentile(y_test, y_pred)]        
    
    return metrics

def encoding_df(df, cols):
    """
    function to encode categorical. 
    BEWARE : do not put all the columns in there otherwise it overloads the RAM and breaks the kernel. Do not use those columns,
    as there is too many unique values :

    sku                               131071
    flight                              3570
    sto                                77649
    aldt                               97252
    eibt                               88465
    cibt                               87972
    aibt                               88146
    chocks_on                          76545

    input: df and columns to encode
    output : encoded df without integer and float columns

    """
    import pandas as pd
    df = df[cols]
    obj_df = df.select_dtypes(include=['object']).copy()
    num_var = df.select_dtypes(include=['int','float']).copy()
    cat_var = pd.get_dummies(obj_df, columns = obj_df.columns)
    encoded_df = pd.concat([num_var, cat_var], axis=1, sort=False)
    return encoded_df

# TEST
# encoding_df(df,['PRCP','TAVG','AWND','TMAX','carrier'])

#----------

def decision_tree(df, variables, test_size):
    """
    function building decision tree

    input: original df, variables to use, test size 
    output : predicted values and rmse

    """
    from sklearn.model_selection import train_test_split
    from sklearn import tree

    # Define input 
    X = encoding_df(df, variables)

    # Set validation
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    clf = tree.DecisionTreeRegressor()
    clf = clf.fit(X_train, y_train)

    print(compute_rmse(y_test, clf.predict(X_test)))
    return clf.predict(X_test), y_test

# TEST
# variables = ['carrier','runway','stand','Manufacturer','PRCP','TAVG','AWND','TMAX','TMIN','WDF2','WDF5','WSF2','WSF5','WT01','WT02','WT03','WT08','Approach Speed\n(Vref)']
# decision_tree(df, variables, 0.10)

# ----------

def linear_reg(df, variables, test_size):
    """
    function building linear regression 

    input: original df, variables to use, test size 
    output : predicted values and rmse

    """
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    # Define input 
    X = encoding_df(df, variables)

    # Set validation
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    clf = LinearRegression().fit(X_train, y_train)

    print('Linear Regression RMSE',compute_rmse(y_test, clf.predict(X_test)))
    return clf.predict(X_test), y_test

# TEST
# variables = ['carrier','runway','stand','Manufacturer','PRCP','TAVG','AWND','TMAX','TMIN','WDF2','WDF5','WSF2','WSF5','WT01','WT02','WT03','WT08','Approach Speed\n(Vref)']
# linear_reg(df, variables,0.10)


def master_modelisation(X,y,test_size,model_function_list) :
    """
    function aggregating models and building comparison table

    input: training features, target values, test_size, list of functions created that are models
    output : table

    """    
    models = model_function_list
    
    output = pd.DataFrame()
    model = []
    RMSE = []
    MAPE = []
    tenth_perc = []    
    ninetieth_perc = []
    
    for i in models:
        output_row = pd.DataFrame()
        
        model.append(i)
        
        y_pred, y_test = i(df, variables, test_size)

        RMSE.append(compute_rmse(y_test, y_pred))
        MAPE.append(compute_mape(y_test, y_pred))
        tenth_perc.append(tenth_percentile(y_test, y_pred))   
        ninetieth_perc.append(ninetieth_percentile(y_test, y_pred))
        
    output['Model'] = model
    output['RMSE'] = RMSE
    output['MAPE'] = MAPE
    output['tenth_perc'] = tenth_perc
    output['ninetieth_perc'] = ninetieth_perc
        
    return output

# TEST
# master_modelisation(X,y,0.10,[linear_reg,decision_tree])