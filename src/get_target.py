import numpy as np
import pandas as pd
import os
import sys
from src import preprocessing
from src import master

def get_target(Path_airport):
    y=preprocessing.get_target_values(
        preprocessing.get_df_of_obs1(
            preprocessing.cleaning_airport_df(Path_airport)
            )
                )
    y.reset_index()
    return y