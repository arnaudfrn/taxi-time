import pandas as pd
from datetime import timedelta  
 
from src.preprocessing import *
from src.master import *



df1 = cleaning_airport_df('/Users/tristanmayer/Desktop/Supercase Eleven/data/0. Airport data/Airport_Data.csv')

df2 = get_df_of_obs1(df1)

df3 = get_target_values(df2)



## Creation des timestamps
## Cleaning of rows that font chier la bite
df1 = df1.loc[df1['aldt']!='aldt']
for col in ['aldt', 'aibt','aobt', 'atot']:
    df1[col] = pd.to_datetime(df1[col])

# number of planes in the airport at time_t (a date in string format)
# number of planes in block at time_t (a date in string format)
#### need to reconcile flights with plane (we only know details about flight so not the same line for arrival and departure from the stand -> see the dates regarding one stand)

# number of planes in movement at time_t (a date in string format)
## input: date as a string
## output: integer
def get_nb_of_planes_in_movement(time_t):
    t = pd.to_datetime(time_t)
    res = len(df1[((df1['aldt']<t) & (df1['aibt']>t))|
                  ((df1['aobt']<t) & (df1['atot']>t))])
    return res

# number of planes that have landed on the runway_R in the last M_min at time_t
## input: runway as a string; date as a string; timedelta as an integer (nb of minutes)
## output: integer
def get_nb_of_planes_runway_in_last_M_min(runway_R, time_t, M_min):
    t1 = pd.to_datetime(time_t)
    t2 = t1-timedelta(minutes = M_min)
    res = len(df1[(df1['runway']==runway_R) &
                  (df1['aldt']<t1) &
                  (df1['aldt']>t2)])
    return res


# number of planes that have stayed at the stand_S in the last M min at time_t
## input: stand as a string, date as a string, timedelta in minutes as float
## output: integer
def get_nb_of_planes_stand_in_last_M_min(stand_S, time_t, M_min):
    t1 = pd.to_datetime(time_t)
    t2 = t1-timedelta(minutes = M_min)
    res = len(df1[(df1['stand']==stand_S) &
                  (df1['aibt']<t1) &
                  (df1['aibt']>t2)])
    return res

# average taxi-time of the planes that have landed on runway_R in the last M min at time_t
## input: runway as a string, date as a string, timedelta in minutes as float
## output: integer (mean)
def average_taxitime_runway_last_X_min(runway_R, time_t, M_min): 
    t1 = pd.to_datetime(time_t)
    t2 = t1-timedelta(minutes = M_min)
    list_of_index = df1[(df1['runway']==runway_R) & 
                        (df1['aibt']<t1) & 
                        (df1['aibt']>t2)].index
    res = df3.loc[list_of_index][0].mean()
    return res

# average taxi-time of the planes that stayed at stand_S in the last M min at time_t
## input: stand as a string, date as a string, timedelta in minutes as float
## output: integer (mean)
def average_taxitime_stand_last_X_min(stand_S, time_t, M_min): 
    t1 = pd.to_datetime(time_t)
    t2 = t1-timedelta(minutes = M_min)
    list_of_index = df1[(df1['stand']==stand_S) & 
                        (df1['aibt']<t1) & 
                        (df1['aibt']>t2)].index
    res = df3.loc[list_of_index][0].mean()
    return res

# taxi_time of the last plane that have landed on runway_R at time_t
## input: runway as a string, date as a string
## output: integer (last taxi time in minutes)
def get_last_taxitime_runway(runway_R, time_t):
    t1 = pd.to_datetime(time_t)
    res = df3.loc[df1[(df1['runway']==runway_R) & (df1['aibt']<t1)]['aibt'].idxmax()][0]
    return res

# taxi_time of the last plane that have stayed at stand_S at time_t
## input: stand as a string, date as a string
## output: integer (last taxi time in minutes)
def get_last_taxitime_stand(stand_S, time_t):
    t1 = pd.to_datetime(time_t)
    res = df3.loc[df1[(df1['stand']==stand_S) & (df1['aibt']<t1)]['aibt'].idxmax()][0]
    return res

    
# taxi_time of the last plane that have gone from runway_R to stand_S at time_t
## input: runway as a string, stand as string, date as a string
## output: integer (last taxi time of the plane that went from runway_R to stand_S)
def get_last_taxitime_path(runway_R, stand_S, time_t):
    t1 = pd.to_datetime(time_t)
    res = df3.loc[df1[(df1['runway']==runway_R) & (df1['stand']==stand_S) & (df1['aibt']<t1)]['aibt'].idxmax()][0]
    return res