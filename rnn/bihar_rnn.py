import sys
import os
sys.path.append('..')

import time

import pickle
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
from sklearn.ensemble import RandomTreesEmbedding, RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd
import re
from utils import *
from constants import *
from rnn import RNN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# def train():
#     start_time = time.time()

#     for epoch in range(NUM_EPOCHS):

#         for i, (X, y) in enumerate(train_loader):
#             X, y = X.type(torch.float32), y.type(torch.float32)
#             # X, y = X.to(device), y.to(device)

#             y_hat = model(X)
#             y_hat = y_hat.squeeze(2)

#             train_loss = torch.sqrt(criterion(y, y_hat))
#             optimizer.zero_grad()
#             train_loss.backward()
#             optimizer.step()

#         train_losses.append(train_loss.item())

#         # if (epoch+1)%5 == 0:
#         print(f'Epoch: {epoch+1} | {NUM_EPOCHS} \t Train Loss: {train_losses[-1]:.4f} \t \
#               Time taken: {(time.time()-start_time)/60:.2f} mins')

if __name__ == '__main__':
    
    device_id = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # torch.cuda.set_device(1)
    print(f'Selected device: {device_id}, device id: {torch.cuda.current_device()}')

    # device_properties = torch.cuda.get_device_properties(device_id)

    # Print the device ID and other information
    # print(f"Current GPU Device ID: {device_id}")
    # print(f"GPU Name: {device_properties.name}")
    # print(f"GPU Capability: {device_properties.major}.{device_properties.minor}")
    # print(f"Total GPU Memory: {device_properties.total_memory / (1024 ** 2):.2f} MB")

    # num_gpus = torch.cuda.device_count()
    # if num_gpus > 0:
    #     print(f"Number of available GPUs: {num_gpus}")
    #     # Print information about each GPU
    #     for i in range(num_gpus):
    #         gpu_name = torch.cuda.get_device_name(i)
    #         print(f"GPU {i}: {gpu_name}")
    # else:
    #     print("No GPU devices found.")

    # with open(f'{data_bihar}/bihar_512_sensor_data_imputed.pkl', "rb") as f:
    #     data = pickle.load(f)
    #     data = data.reset_index()
    #     data = data[['timestamp', 'latitude', 'longitude', 'rh', 'temp', 'pm25']]
    
    # data['rh'] = (data['rh']-data['rh'].mean()) / data['rh'].std()
    # data['temp'] = (data['temp']-data['temp'].mean()) / data['temp'].std()
    # data['meteo'] = data.apply(lambda row: [row['rh'], row['temp']], axis=1)

    # df = data[['timestamp', 'latitude', 'longitude', 'meteo', 'pm25']]
    # c_map = {'timestamp': 'Timestamp', 'latitude': 'Latitude', 'longitude': 'Longitude', 'meteo': 'Meteo', 'pm25': 'PM25'}
    # df = df.rename(columns=c_map)

    # cols = {'Timestamp': 'datetime64[ns]', 'Latitude': np.float32, 'Longitude': np.float32, 'PM25': np.float32}
    # train_df = df.astype(cols)

    # station_indexing_train = station_indexing(train_df)
    # data_train = create_timeseries_data(train_df, station_indexing_train)
    # train_dataset = TimeSeriesDataset(data=data_train)

    # BATCH_SIZE = 1
    # LEARNING_RATE = 1e-4
    # INPUT_DIM = len(data_train[0][0]['Meteo'])
    # HIDDEN_DIM = 64
    # LAYER_DIM = 1
    # NUM_EPOCHS = 5
    # TYPE = 'LSTM'
    # BIDIRECTIONAL = True

    # model = RNN(TYPE, INPUT_DIM, LAYER_DIM, HIDDEN_DIM, BIDIRECTIONAL, device)
    # model.to(device)

    # criterion = nn.MSELoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    # train_losses = []

    # train()