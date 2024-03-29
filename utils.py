import math
import random
import pickle
import time
import numpy as np
import pandas as pd
from scipy import stats
from xgboost import XGBRegressor
from constants import *
from sklearn.ensemble import RandomTreesEmbedding, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset

random.seed(42)

''' Create a dictionary of train, validation and test stations using latitude and longitude informations
    Parameters:
        stations: A list of tuple (latitude, longitude)
'''
def lat_long_split_stations(stations, loc_min, loc_max, split_ratio):

    stations = [loc for loc in stations if loc != loc_min and loc != loc_max]
    random.shuffle(stations)

    test_index = int((split_ratio[0] + split_ratio[1]) * len(stations))
    val_index = int(split_ratio[0] * len(stations))
    train_stations, val_stations, test_stations = stations[:val_index], stations[val_index:test_index], stations[test_index:]

    train_stations.extend([loc_min, loc_max])

    return train_stations, val_stations, test_stations

def load_locs_as_tuples(file_path):
    tuples_list = []

    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into two values and convert them to floats
            values = line.strip().split()
            tuple_values = (float(values[0]), float(values[1]))
            
            # Append the tuple to the list
            tuples_list.append(tuple_values)

    return tuples_list

'''
    Define a custom upper and lower limit for XGBoost predictions
'''
def custom_eval_metric(y_true, y_pred):
    y_pred = np.clip(y_pred, LOWER_BOUND, UPPER_BOUND)
    return 'custom_eval_metric', np.mean(np.abs(y_true - y_pred))

'''

'''
def train_test_split(df, cols, split_ratio=[0.4, 0.1, 0.5], split_type='lat_long', normalize=True, load_locs=True):

    assert split_type in ({'lat_long', 'timestamp'}), "Wrong split type"
    assert sum(split_ratio) == 1.0, "Wrong split ratio provided"

    df['timestamp'] = df['timestamp'].values.astype(float)
    df['pm25'] = df['pm25'].values.astype(float)
    df = df[cols]

    c = 'loc' if split_type == 'lat_long' else 'ts'

    if split_type == 'lat_long':
        df[c] = list(zip(df['latitude'], df['longitude']))
        loc_min, loc_max = df.nsmallest(1, 'pm25')[c].iloc[0], df.nlargest(1, 'pm25')[c].iloc[0]
        locs = df[c].unique()
        if load_locs:
            train_idxs = load_locs_as_tuples(f'{data_bihar}/train_locations.txt')
            val_idxs = load_locs_as_tuples(f'{data_bihar}/val_locations.txt')
            test_idxs = load_locs_as_tuples(f'{data_bihar}/test_locations.txt')
        else:
            train_idxs, val_idxs, test_idxs = lat_long_split_stations(locs, loc_min, loc_max, split_ratio)
            np.savetxt(f'{data_bihar}/train_locations.txt', train_idxs, fmt='%f')
            np.savetxt(f'{data_bihar}/val_locations.txt', val_idxs, fmt='%f')
            np.savetxt(f'{data_bihar}/test_locations.txt', test_idxs, fmt='%f')
    else:
        df[c] = df['timestamp']
        ts = sorted(df[c].unique())
        test_index = int((split_ratio[0] + split_ratio[1]) * len(ts))
        val_index = int(split_ratio[0] * len(ts))
        train_idxs, val_idxs, test_idxs = ts[:val_index], ts[val_index:test_index], ts[test_index:]

    if normalize:
        scaler = StandardScaler()
        data = df[[x for x in cols if x != 'pm25']].to_numpy()
        data = scaler.fit_transform(data)
        df[[x for x in cols if x != 'pm25']] = data
    
    train_df = df[df[c].isin(train_idxs)]
    train_df = train_df[cols]

    val_df = df[df[c].isin(val_idxs)]
    val_df = val_df[cols]

    test_df = df[df[c].isin(test_idxs)]
    test_df = test_df[cols]

    train_data, val_data, test_data = train_df.to_numpy(), val_df.to_numpy(), test_df.to_numpy()
    X_train, y_train, X_val, y_val, X_test, y_test = train_data[:, :-1], train_data[:, -1], val_data[:, :-1], val_data[:, -1],\
        test_data[:, :-1], test_data[:, -1]

    return X_train, y_train, X_val, y_val, X_test, y_test

''' Create pandas dataframe from pickle file
    Input:
        List with each data point as dictionary with Timestamp, Meteo, Image and PM2.5 information
        Image: Resized to 224*224*3
        Meteo: RH, Temp, BP, Latitude and Longitude Information
    Output:
        Dataframe with Timestamp, Latitude, Longitude, Meteorological and PM2.5 information
'''
def create_dataframe(data, latlong=False):

    df = pd.DataFrame(data)
    # cols = ['Timestamp', 'Latitude', 'Longitude', 'Meteo', 'PM25']

    df['Latitude'] = df['Meteo'].apply(lambda x: x[-2])
    df['Longitude'] = df['Meteo'].apply(lambda x: x[-1])
    if latlong == False:
        df['Meteo'] = df['Meteo'].apply(lambda x: x[:-2])
    
    df = df[['Timestamp', 'Latitude', 'Longitude', 'Meteo', 'PM25']]
    
    return df

''' Assign each station (Latitude, Longitude) a unique index
    Input:
        A pandas dataframe
    Output:
        A dictionary with Latitude and Longitude information as key and a unique index as corresponding value
'''
def station_indexing(data):
    
    station_indexing = {}

    keys = data.groupby(['Latitude', 'Longitude']).groups.keys()
    idxs = [i for i in range(len(keys))]

    for i, key in zip(idxs, keys):
        station_indexing[key] = i
    
    return station_indexing

''' Input:
        df: A pandas dataframe with Timestamp, Latitude, Longitude, Meteo and PM2.5 information
        station_indexing: A dictionary with Latitude and Longitude information as key and a unique index as corresponding value
    Output:
        Create an ordered timeseries dataset, with a single datapoint containing ordered Meteo, PM2.5 information for a single
        station
'''
def create_timeseries_data(df, station_indexing):

    df = df.sort_values(['Timestamp'])

    keys = df.groupby(['Latitude', 'Longitude']).groups.keys()
    data = [[] for _ in range(len(keys))]

    for _, row in df.iterrows():
        key = (row['Latitude'], row['Longitude'])
        data[station_indexing[key]].append({'Meteo': row['Meteo'], 'PM25': row['PM25']})

    return data

''' Input:
        y_pred: Model predictions
        y: Actual labels
    Output:
        Root Mean Square Value, Spearman R_squared, Spearman p_value, Pearson R_squared, Pearson p_value
'''
def eval_stat(y_pred, y):
    RMSE = math.sqrt(mean_squared_error(y_pred, y))
    R_squared = stats.spearmanr(y_pred, y.ravel())[0]
    p_value = stats.spearmanr(y_pred, y.ravel())[1]
    R_squared_pearson = stats.pearsonr(y_pred, y.ravel())[0]
    p_value_pearson = stats.pearsonr(y_pred, y.ravel())[1]
    return RMSE, R_squared, p_value, R_squared_pearson, p_value_pearson

''' Change input data to a sparse embedding 
    Input:
        X: Data without labels -> Shape: (N, d)
    Output:
        Sparse embedding of maximum shape (N, n_estimators*max_depth**2)        
'''
def random_tree_embedding(X, n_estimators, max_depth):
    rt_model = RandomTreesEmbedding(n_estimators=n_estimators, max_depth=max_depth).fit(X)
    X_transformed = rt_model.transform(X).toarray()
    return X_transformed

''' Train a random forest regressor
    Input:
        X: Sparse embedding of data (as learnt from random_tree_embedding)
        y: PM2.5 labels
    Output:
        y_pred: PM2.5 predictions
'''
def random_forest_regressor(X, y, n_estimators, min_samples_leaf):
    rf_model = RandomForestRegressor(n_estimators=n_estimators, max_features="sqrt", min_samples_leaf=min_samples_leaf).fit(X, y)
    y_pred = rf_model.predict(X)
    return y_pred

'''
    Get the performance of our custom XGBoost model
'''
def train_XGBoost(X_train, y_train, X_val, y_val, X_test, y_test, **model_args):
    model = XGBRegressor(objective ='reg:squarederror', eval_metric=custom_eval_metric)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val) if X_val.shape[0] != 0 else None
    y_test_pred = model.predict(X_test)
    
    train_stat = eval_stat(y_train_pred, y_train)
    val_stat = eval_stat(y_val_pred, y_val) if X_val.shape[0] != 0 else [None] * 5
    test_stat = eval_stat(y_test_pred, y_test)
    
    if len(model_args) != 0:
        file_name = model_args['model_name']
        pickle.dump(model, open(f'{model_dir}/{file_name}.pkl', 'wb'))

    stats = {'Train_RMSE': train_stat[0], 'Train_Pearson_R': train_stat[1],\
          'Val_RMSE': val_stat[0], 'Val_Pearson_R': val_stat[1], 'Test_RMSE': test_stat[0], 'Test_Pearson_R': test_stat[1]}
    
    return stats

def data_processing(df, train_locs, val_locs, test_locs, WS, FW):
    df_grouped = df.groupby(['latitude', 'longitude'])
    train_data, val_data, test_data = [], [], []

    start_time = time.time()
    print(f"---------\t Dataset processing started; FORECAST WINDOW = {FW}\t---------")

    for loc, group in df_grouped:

        data = group.to_numpy()
        # Since first three columns are timestamp, latitude and longitude respectively
        X, y = data[:, 3:-1], data[:, -1]

        '''
            Vectorized code for making different windows of data
        '''
        y = np.lib.stride_tricks.sliding_window_view(y, (FW,))
        X = X[:y.shape[0], :]
        X = np.lib.stride_tricks.sliding_window_view(X, (WS, X.shape[1]))
        y = np.lib.stride_tricks.sliding_window_view(y, (WS, y.shape[1]))
        X, y = np.squeeze(X), np.squeeze(y)

        if FW == 1:
            y = y.reshape(y.shape[0], -1, 1)

        X, y = X.astype(np.float32), y.astype(np.float32)

        assert X.shape[0] == y.shape[0] and X.shape[1] == y.shape[1]

        # data = [{'meteo': X_w.astype(np.float32), 'pm25': y_w.astype(np.float32)} for X_w, y_w in zip(X, y)]
        if loc in train_locs:
            train_data.extend([{'meteo': X_w, 'pm25': y_w} for X_w, y_w in zip(X, y)])
        elif loc in val_locs:
            val_data.extend([{'meteo': X_w, 'pm25': y_w} for X_w, y_w in zip(X, y)])
        elif loc in test_locs:
            test_data.extend([{'meteo': X_w, 'pm25': y_w} for X_w, y_w in zip(X, y)])
        
    print("---------\t Dataset processing completed \t---------")
    print(f'Time taken: {(time.time()-start_time)/60:.2f} mins')

    return train_data, val_data, test_data