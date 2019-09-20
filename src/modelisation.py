{\rtf1\ansi\ansicpg1252\cocoartf1671\cocoasubrtf500
{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fmodern\fcharset0 Courier;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;\red255\green255\blue255;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;\cssrgb\c100000\c100000\c100000;}
\paperw11900\paperh16840\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 \
# \'97\'97\'97 Xav \'97\'97\'97\
\
def compute_rmse(y, y_pred)\
    """\
    function computing RMSE \
\
    input: real target values and predicted values\
    output : root mean squared error in minutes\
    """\
    return np.sqrt((y - y_pred)**2).mean() / 60\
\
def linear_reg(X,y):\
    """\
    function building linear regression \
\
    input: all features to use for modelling and real target values\
    output : predicted values and print OLS summary\
\
    """\
\
    from sklearn.linear_model import LinearRegression\
    import statsmodels.api as sm\
    \
    reg = LinearRegression().fit(X, y)\
    model = sm.OLS(y, X)\
    results = model.fit()\
    y_pred = reg.predict(X)\
    \
    # statsmodel gives a good summary of model performance\
    print(results.summary())\
    return y_pred\
\
def encoding_categorical(df, col_to_encode):\
        """\
    function to encode categorical. \
    BEWARE : do not put all the columns in there otherwise it overloads the RAM and breaks the kernel. Do not use those columns, as there is too many unique values :\
\
\pard\pardeftab720\sl320\partightenfactor0

\f1\fs28 \cf2 \cb3 \expnd0\expndtw0\kerning0
sku                               131071\
flight                              3570\
sto                                77649\
aldt                               97252\
eibt                               88465\
cibt                               87972\
aibt                               88146\
chocks_on                          76545
\f0\fs24 \cf0 \cb1 \kerning1\expnd0\expndtw0 \
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0
\cf0 \
    input: df and columns to encode\
    output : encoded df without integer and float columns\
\
    """\
    obj_df = df.select_dtypes(include=['object']).copy()\
    encoded_df = pd.get_dummies(obj_df, columns = col_to_encode)\
    return encoded_df.select_dtypes(exclude =['object'])\
\
}