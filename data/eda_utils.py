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
def impute(data, method):
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


''' Get region wise RMSE, Pearson R values
    Parameters:
        df: A 2D numpy array with each row corresponding to Timestamp, Latitude, Longitude, RH, Temp, PM2.5 values (in this order)
        model_dir: Directory in which the trained model needs to be stored
        split_type: The split to be applied between train and test data (random, lat_long, timestamp forecasting)
        model_type: The regression model to be used for PM2.5 prediction (rt_rf, xg_boost)
        include_latlong: Whether to add lat_long information while predicting the PM2.5 values
        include_timestamp: Whether to add timestamp information while predicting the PM2.5 values
'''
def train_and_eval(data, model_dir, method, split='lat_long', model_type='xgb'):
    
    assert split == 'random' or split == 'lat_long' or split == 'timestamp', \
    'split can only be random, lat_long or timestamp'
    assert model_type == 'rt_rf' or model_type == 'xgb', 'model_type can only be rt_rf or xgb'

    stat_data = []

    X_train, X_test, y_train, y_test = [], [], [], []

    if split == 'random':
        X_train, X_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.33)

    elif split == 'latlong':
        train_stations, test_stations = lat_long_split_stations(list(set(zip(data[:, 1], data[:, 2]))))

        for lat_long, data in zip(data[:, 1:3], data):
            if tuple(lat_long) in train_stations:
                X_train.append(data[:-1])
                y_train.append(data[-1])
            elif tuple(lat_long) in test_stations:
                X_test.append(data[:-1])
                y_test.append(data[-1])
    
    else:
        train_timestamps, test_timestamps = timestamp_split(set(data[:, 0]))
        for ts, data in zip(data[:, 0], data):
                if ts in train_timestamps:
                    X_train.append(data[:-1])
                    y_train.append(data[-1])
                elif ts in test_timestamps:
                    X_test.append(data[:-1])
                    y_test.append(data[-1])

    
    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    if model_type == 'rt_rf':
        model = RandomTreesEmbedding(n_estimators=800,max_depth=2).fit(X_train)
        X_train = model.transform(X_train).toarray()
        X_test = model.transform(X_test).toarray()

        model = RandomForestRegressor(n_estimators=800, max_features="sqrt", min_samples_leaf=2).fit(X_train, y_train)
    elif model_type == 'xgb':
        model = XGBRegressor(objective ='reg:squarederror', eval_metric=custom_eval_metric)
        model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_stat = eval_stat(y_train_pred, y_train)
    test_stat = eval_stat(y_test_pred, y_test)

    stat_data.append({'Train_RMSE': train_stat[0], 'Train_Pearson_R': train_stat[1], \
                    'Test_RMSE': test_stat[0], 'Test_Pearson_R': test_stat[1]})
        
    model_name = model_dir + f"bihar_{model_type}_{method}_{split}.pkl"
    joblib.dump(model, model_name)
    
    return stat_data

''' Create a dictionary of train and test stations using latitude and longitude informations
    Parameters:
        df: A pandas dataframe with columns Timestamp, Region, Latitude, Longitude, Meteo, PM2.5 information (in this order) 
'''
def lat_long_split_stations(stations):

    random.shuffle(stations)

    index = int(len(stations)/1.5)
    train_stations, test_stations = set(stations[:index]), set(stations[index:])

    return train_stations, test_stations

''' Create a dictionary of train and test timestamp values
    Parameters:
        timestamps: A python set containing all the unique timestamp values for a particular region
'''
def timestamp_split(timestamps):
    timestamps = list(timestamps)
    sorted(timestamps)
    index = len(timestamps)//2
    train_timestamps, test_timestamps = set(timestamps[:index]), set(timestamps[index:])
    return train_timestamps, test_timestamps

def custom_eval_metric(y_true, y_pred):
    lower_bound = 0
    upper_bound = 500
    y_pred = np.clip(y_pred, lower_bound, upper_bound)
    return 'custom_eval_metric', np.mean(np.abs(y_true - y_pred))