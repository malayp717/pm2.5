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

class InvalidImputerException(Exception):
    "Imputer type can only be KNN, Mean or Iterative"
    pass


''' Use inbuilt sklearn functions to fill the missing nan values in the data '''
def impute(data, method):
    # KNN Imputer
    if method == 'knn':
        imputer = KNNImputer(n_neighbors=2)
    # Mean Imputer
    elif method == 'mean':
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    # Iterative Imputer
    elif method == 'iterative':
        imputer = IterativeImputer(random_state=0)
    else:
        raise InvalidImputerException
    
    return imputer.fit_transform(data)


''' Get region wise RMSE, Pearson R values '''
def region_wise_stat(df, method='knn', lat_long_split=False, include_timestamp=True, drop_nan=False):

    if drop_nan:
        df = df.dropna(subset=['PM25'])

    grp = df.groupby('Region')
    stat_data = []

    if lat_long_split:
        train_stations, test_stations = lat_long_split_stations(df)

    for name, group in grp:
        grp_data = []

        for _, data in group.iterrows():
            row = []
            if include_timestamp:
                date = dateutil.parser.parse(data['Timestamp'].strftime('%Y-%m-%d %X'))
                row.append(date.timestamp())
            row.extend(data['Meteo'])
            row.append(data['PM25'])
            grp_data.append(row)
        
        grp_data = np.array(grp_data)
        imputed_data = impute(grp_data, method=method)

        if not lat_long_split:
            X, y = imputed_data[:, :-1], imputed_data[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        else:
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

        # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        rt_model = RandomTreesEmbedding(n_estimators=800,max_depth=2).fit(X_train)
        data_transformed_train = rt_model.transform(X_train).toarray()
        data_transformed_test = rt_model.transform(X_test).toarray()

        rf_model = RandomForestRegressor(n_estimators=800, max_features="sqrt", min_samples_leaf=2).fit(data_transformed_train, y_train)
        y_train_pred_rf = rf_model.predict(data_transformed_train)
        y_test_pred_rf = rf_model.predict(data_transformed_test)

        train_stat = eval_stat(y_train_pred_rf, y_train)
        test_stat = eval_stat(y_test_pred_rf, y_test)

        stat_data.append({'Region': name, 'Train_RMSE': train_stat[0], 'Train_Pearson_R': train_stat[1], \
                        'Test_RMSE': test_stat[0], 'Test_Pearson_R': test_stat[1]})
    
    return stat_data

''' Get region wise RMSE, Pearson R values '''
def region_wise_stat_xgboost(df, method='knn', lat_long_split=False, include_timestamp=True, drop_nan=False):

    if drop_nan:
        df = df.dropna(subset=['PM25'])

    grp = df.groupby('Region')
    stat_data = []

    if lat_long_split:
        train_stations, test_stations = lat_long_split_stations(df)

    for name, group in grp:
        grp_data = []

        for _, data in group.iterrows():
            row = []
            if include_timestamp:
                date = dateutil.parser.parse(data['Timestamp'].strftime('%Y-%m-%d %X'))
                row.append(date.timestamp())
            row.extend(data['Meteo'])
            row.append(data['PM25'])
            grp_data.append(row)
        
        grp_data = np.array(grp_data)
        imputed_data = impute(grp_data, method=method)

        if not lat_long_split:
            X, y = imputed_data[:, :-1], imputed_data[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        else:
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
        
        xgb_r = XGBRegressor(objective ='reg:squarederror')
        xgb_r.fit(X_train, y_train)

        train_stat = eval_stat(xgb_r.predict(X_train), y_train)
        test_stat = eval_stat(xgb_r.predict(X_test), y_test)

        stat_data.append({'Region': name, 'Train_RMSE': train_stat[0], 'Train_Pearson_R': train_stat[1], \
                        'Test_RMSE': test_stat[0], 'Test_Pearson_R': test_stat[1]})
        
    return stat_data


def lat_long_split_stations(df):
    grps = df.groupby(['Latitude', 'Longitude'])
        
    stations = []

    for key, _ in grps:
        stations.append(key)
    random.shuffle(stations)

    index = int(len(stations)/1.5)
    train_stations, test_stations = set(stations[:index]), set(stations[index:])

    return train_stations, test_stations