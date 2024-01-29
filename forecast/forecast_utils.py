import sys
import os
sys.path.append('..')
import time
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
from constants import *

if __name__ == '__main__':

    data_file = f'{data_bihar}/bihar_512_sensor_era5_rnn.pkl'
    df = pd.read_pickle(data_file)

    scaler = StandardScaler()
    data = df[[x for x in df.columns if x not in {'timestamp', 'latitude', 'longitude', 'pm25'}]].to_numpy()
    data = scaler.fit_transform(data)
    df[[x for x in df.columns if x not in {'timestamp', 'latitude', 'longitude', 'pm25'}]] = data

    # print(df.head(100))


    df_grouped = df.groupby(['latitude', 'longitude'])
    dataset = {}

    for loc, group in df_grouped:

        data = group.to_numpy()
        X, y = data[:, 3:-1], data[:, -1]

        y = np.lib.stride_tricks.as_strided(y, shape=(y.size - WINDOW_SIZE + 1, WINDOW_SIZE), strides=(y.strides[0], y.strides[0]))
        X = X[:y.shape[0], :]

        row = {'meteo': X, 'pm25': y}
        print(X.shape, y.shape)
        dataset[loc] = row
    
    # print(dataset)