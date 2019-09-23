# ——— Xav ———

def compute_rmse(y, y_pred)
    """
    function computing RMSE 

    input: real target values and predicted values
    output : root mean squared error in minutes
    """
    return np.sqrt(((y_pred - y)**2).mean()) / 60

def linear_reg(X,y):

    from sklearn.linear_model import LinearRegression
    import statsmodels.api as sm
    
    reg = LinearRegression().fit(X, y)
    model = sm.OLS(y, X)
    results = model.fit()
    y_pred = reg.predict(X)
    
    # statsmodel gives a good summary of model performance
    print(results.summary())
    return y_pred

def encoding_df(df, cols):
        """
    function to encode categorical. 
    BEWARE : do not put all the columns in there otherwise it overloads the RAM and breaks the kernel. Do not use those columns, as there is too many unique values :

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
    df = df[cols].copy()
    obj_df = df.select_dtypes(include=['object']).copy()
    num_var = df.select_dtypes(include=['int','float']).copy()
    cat_var = pd.get_dummies(obj_df, columns = obj_df.columns)
    encoded_df = pd.concat([num_var, cat_var], axis=1, sort=False)
    return encoded_df

# TEST
# encoding_df(df,['PRCP','TAVG','AWND','TMAX','carrier'])

----------

def decision_tree(df, variables,test_size):
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
    return clf.predict(X_test)

# TEST
# variables = ['carrier','runway','stand','Manufacturer','PRCP','TAVG','AWND','TMAX','TMIN','WDF2','WDF5','WSF2','WSF5','WT01','WT02','WT03','WT08','Approach Speed\n(Vref)']
# decision_tree(df, variables,0.10)

----------

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

    reg = LinearRegression().fit(X_train, y_train)

    print(compute_rmse(y_test, reg.predict(X_test)))
    return reg.predict(X_test)

# TEST
# variables = ['carrier','runway','stand','Manufacturer','PRCP','TAVG','AWND','TMAX','TMIN','WDF2','WDF5','WSF2','WSF5','WT01','WT02','WT03','WT08','Approach Speed\n(Vref)']
# linear_reg(df, variables,0.10)