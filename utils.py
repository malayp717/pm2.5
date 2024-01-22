import math
import random
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from xgboost import XGBRegressor
from sklearn.ensemble import RandomTreesEmbedding, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset

''' Create a dictionary of train and test stations using latitude and longitude informations
    Parameters:
        stations: A list of tuple (latitude, longitude)
'''
def lat_long_split_stations(stations):

    random.shuffle(stations)
    index = int(len(stations)/1.5)
    train_stations, test_stations = set(stations[:index]), set(stations[index:])

    return train_stations, test_stations

'''
    Define a custom upper and lower limit for XGBoost predictions
'''
def custom_eval_metric(y_true, y_pred):
    lower_bound = 0
    upper_bound = 600
    y_pred = np.clip(y_pred, lower_bound, upper_bound)
    return 'custom_eval_metric', np.mean(np.abs(y_true - y_pred))

'''

'''
def train_test_split(df, cols, split_type='lat_long', normalize=True):

    assert split_type in ({'lat_long', 'timestamp'}), "Wrong split type"

    df['timestamp'] = df['timestamp'].values.astype(float)
    df['pm25'] = df['pm25'].values.astype(float)
    df = df[cols]

    c = 'loc' if split_type == 'lat_long' else 'ts'

    if split_type == 'lat_long':
        df[c] = list(zip(df['latitude'], df['longitude']))
        locs = df[c].unique()
        train_idxs, test_idxs = lat_long_split_stations(locs)
    else:
        df[c] = df['timestamp']
        ts = sorted(df[c].unique())
        x = int(len(ts)/1.5)
        train_idxs, test_idxs = ts[:x], ts[x:]

    if normalize:
        scaler = StandardScaler()
        data = df[[x for x in cols if x != 'pm25']].to_numpy()
        data = scaler.fit_transform(data)
        df[[x for x in cols if x != 'pm25']] = data
    
    train_df = df[df[c].isin(train_idxs)]
    train_df = train_df[cols]

    test_df = df[df[c].isin(test_idxs)]
    test_df = test_df[cols]

    train_data, test_data = train_df.to_numpy(), test_df.to_numpy()
    X_train, y_train, X_test, y_test = train_data[:, :-1], train_data[:, -1], test_data[:, :-1], test_data[:, -1]

    return X_train, y_train, X_test, y_test

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
def train_XGBoost(X_train, y_train, X_test, y_test):
    model = XGBRegressor(objective ='reg:squarederror', eval_metric=custom_eval_metric)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_stat = eval_stat(y_train_pred, y_train)
    test_stat = eval_stat(y_test_pred, y_test)

    print(f'Train_RMSE: {train_stat[0]},\t Train_Pearson_R: {train_stat[1]},\t \
            Test_RMSE: {test_stat[0]},\t Test_Pearson_R: {test_stat[1]}')

''' Convert the timeseries dataset to PyTorch timeseries dataset
    Input: Time Ordered Data for each station
        A Single Station contains ordered data as a list of dictionaries with
            Meteo: Sparse Random Tree Embedding (as learnt from random_tree_embedding)
            PM2.5: Corresponding PM2.5 labels
    Output: Same data converted to a PyTorch tensor
'''
class TimeSeriesDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        X, y = [], []

        for item in self.data[idx]:
            X.append(item['Meteo'])
            y.append(item['PM25'])

        return np.array(X), np.array(y)