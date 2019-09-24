import pandas as pd
from datetime import timedelta  
import os
from src.preprocessing import *
from src.master import *


#------------------ Tristan ------------------------
#### Functions to get a proper split of the training and testing dataset:

## function 'merge_df' to apply after the master_preprocessing_X function ran on both the test and train csv
### input: two dataframes IN ORDER: train and THEN test
### output: a merged dataframe ready to be processed by the features_pimpage function
def merge_df(df_train_clean, df_test_clean):
	df_train_clean['TESTING'] = 0
	df_test_clean['TESTING'] = 1
	df_tot = pd.concat([df_train_clean, df_test_clean]).reset_index(drop=True)
	return df_tot




## function 'unmerge_df' to apply after the features_pimpage function ran on the merged dataframe 
### input: dataframe transformed by features_pimpage
### output: two dataframes, two targets dataframes
def unmerged_df(df_tot):
	X_train = df_tot[0][df_tot[0]['TESTING']==0].drop('TESTING', axis=1)
	y_train = df_tot[1].loc[X_train.index]
	X_test = df_tot[0][df_tot[0]['TESTING']==1].drop('TESTING', axis=1)
	y_test = df_tot[1].loc[X_test.index]
	return X_train, y_train, X_test, y_test