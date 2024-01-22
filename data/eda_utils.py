import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomTreesEmbedding, RandomForestRegressor
from utils import eval_stat
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
import dateutil.parser
import impyute
import random
import xgboost
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import joblib

''' Use inbuilt sklearn functions to fill the missing nan values in the data '''
def impute(data, method='iterative'):
    assert method == 'knn' or method == 'mean' or method == 'iterative', 'method can only knn, mean or iterative'
    # KNN Imputer
    if method == 'knn':
        imputer = KNNImputer(n_neighbors=2)
    # Mean Imputer
    elif method == 'mean':
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    # Iterative Imputer
    elif method == 'iterative':
        imputer = IterativeImputer(random_state=0)
    
    return imputer.fit_transform(data)

def find_distance(x, y):
    return np.abs(x[0] - y[0]) + np.abs(x[1] - y[1])

''' ERA5 file resolution is different to the original dataset resolution, find the closest location and return
    Parameters:
        orig_df: Original Dataframe
        params_df: ERA5 dataframe
'''
def transform_lat_long(orig_df, params_df):
    lat_lon_orig = list(orig_df.groupby(['latitude', 'longitude']).groups.keys())
    lat_lon_params = list(params_df.groupby(['latitude', 'longitude']).groups.keys())

    # Lat_Lon in the NetCDRF file -> Lat_Lon in the original dataset
    close_locs = {}

    for loc_params in lat_lon_params:
        dist = 10**9
        for loc_orig in lat_lon_orig:
            if find_distance(loc_orig, loc_params) < dist:
                dist = find_distance(loc_orig, loc_params)
                close_locs[loc_params] = loc_orig

    transformed_locs = [close_locs[(x, y)] for (x, y) in zip(params_df['latitude'], params_df['longitude'])]
    
    return map(list, zip(*transformed_locs))