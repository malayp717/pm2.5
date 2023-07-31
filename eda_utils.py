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
        df: A pandas dataframe with columns Timestamp, Region, Latitude, Longitude, Meteo, PM2.5 information (in this order)
        method: The data imputation method to be applied (knn, mean, iterative)
        split_type: The split to be applied between train and test data (random, lat_long, timestamp forecasting)
        model_type: The regression model to be used for PM2.5 prediction (rt_rf, xg_boost)
        include_latlong: Whether to add lat_long information while predicting the PM2.5 values
        include_timestamp: Whether to add timestamp information while predicting the PM2.5 values
'''
def region_wise_stat(df, method='knn', split_type='lat_long', model_type='xg_boost', include_latlong=True,\
                     include_timestamp=True):
    
    assert split_type == 'random' or split_type == 'lat_long' or split_type == 'timestamp', \
    'split_type can only be random, lat_long or timestamp'
    assert model_type == 'rt_rf' or model_type == 'xg_boost', 'model_type can only be rt_rf or xg_boost'

    df = df.dropna(subset=['PM25'])
    grp = df.groupby('Region')
    stat_data = []

    if split_type == 'lat_long':
        train_stations, test_stations = lat_long_split_stations(df)

    for name, group in grp:
        grp_data = []

        if split_type == 'timestamp':
            timestamps = []

        for _, data in group.iterrows():
            row = []
            if include_timestamp:
                date = dateutil.parser.parse(data['Timestamp'].strftime('%Y-%m-%d %X'))
                row.append(date.timestamp())
                if split_type == 'timestamp':
                    timestamps.append(date.timestamp())
            if include_latlong:
                row.extend(data['Meteo'])
            else:
                row.extend(data['Meteo'][:-2])
            row.append(data['PM25'])
            grp_data.append(row)
        
        grp_data = np.array(grp_data)
        imputed_data = impute(grp_data, method=method)

        if split_type == 'random':
            X, y = imputed_data[:, :-1], imputed_data[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        elif split_type == 'lat_long':
            X_train, X_test, y_train, y_test = [], [], [], []
            for data in imputed_data:
                lat_long = (data[-3], data[-2])
                if lat_long in train_stations:
                    X_train.append(data[:-1])
                    y_train.append(data[-1])
                elif lat_long in test_stations:
                    X_test.append(data[:-1])
                    y_test.append(data[-1])
            X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
        elif split_type == 'timestamp':
            train_timestamps, test_timestamps = timestamp_split(set(timestamps))
            X_train, X_test, y_train, y_test = [], [], [], []
            for data in imputed_data:
                time = data[0]
                if time in train_timestamps:
                    X_train.append(data[:-1])
                    y_train.append(data[-1])
                elif time in test_timestamps:
                    X_test.append(data[:-1])
                    y_test.append(data[-1])
            X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

        if model_type == 'rt_rf':
            model = RandomTreesEmbedding(n_estimators=800,max_depth=2).fit(X_train)
            X_train = model.transform(X_train).toarray()
            X_test = model.transform(X_test).toarray()

            model = RandomForestRegressor(n_estimators=800, max_features="sqrt", min_samples_leaf=2).fit(X_train, y_train)
        elif model_type == 'xg_boost':
            model = XGBRegressor(objective ='reg:squarederror')
            model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_stat = eval_stat(y_train_pred, y_train)
        test_stat = eval_stat(y_test_pred, y_test)

        stat_data.append({'Region': name, 'Train_RMSE': train_stat[0], 'Train_Pearson_R': train_stat[1], \
                        'Test_RMSE': test_stat[0], 'Test_Pearson_R': test_stat[1]})
    
    return stat_data

''' Create a dictionary of train and test stations using latitude and longitude informations
    Parameters:
        df: A pandas dataframe with columns Timestamp, Region, Latitude, Longitude, Meteo, PM2.5 information (in this order) 
'''
def lat_long_split_stations(df):
    grps = df.groupby(['Latitude', 'Longitude'])
        
    stations = []

    for key, _ in grps:
        stations.append(key)
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