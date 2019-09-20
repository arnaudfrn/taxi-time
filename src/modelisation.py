{\rtf1\ansi\ansicpg1252\cocoartf1671\cocoasubrtf500
{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 \
# \'97\'97\'97 Xav \'97\'97\'97\
\
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0
\cf0 def compute_rmse(y, y_pred)\
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
}