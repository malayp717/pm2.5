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
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import argparse

import warnings
warnings.filterwarnings('ignore')

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        x, y = sample['meteo'], sample['pm25']
        return torch.Tensor(x), torch.Tensor(y)
    
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, out_dim):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(hidden_dim*2, out_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.gelu(out)
        out = self.fc(out)

        out = out[:, -1, :]
        out = torch.clamp(out, LOWER_BOUND, UPPER_BOUND)
        return out

class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, y_pred, y):
        mse = nn.MSELoss()
        return torch.sqrt(mse(y_pred, y))
    
class FrobeniusNorm(nn.Module):
    def __init__(self):
        super(FrobeniusNorm, self).__init__()
    
    def forward(self, y, y_pred):
        return torch.sqrt(torch.mean(torch.norm(y_pred-y, dim=1)))
    
def data_processing(df, train_locs, val_locs, test_locs, WINDOW_SIZE, FORECAST_WINDOW):
    df_grouped = df.groupby(['latitude', 'longitude'])
    train_data, val_data, test_data = [], [], []

    start_time = time.time()
    print(f"---------\t Dataset processing started; FORECAST WINDOW = {FORECAST_WINDOW}\t---------")

    for loc, group in df_grouped:

        data = group.to_numpy()
        # Since first three columns are timestamp, latitude and longitude respectively
        X, y = data[:, 3:-1], data[:, -1]

        # if FORECAST_WINDOW != 1:
        y_window = []

        for i in range(X.shape[0]-FORECAST_WINDOW):
            y_window.append(y[i: i+FORECAST_WINDOW])

        y_window = np.array(y_window)
        # y = np.lib.stride_tricks.as_strided(y, shape=(y.size - FORECAST_WINDOW + 1, FORECAST_WINDOW), strides=(y.strides[0], y.strides[0]))
        X = X[:y_window.shape[0], :]

        y = y_window

        for i in range(X.shape[0]-WINDOW_SIZE):
            X_w, y_w = X[i: i+WINDOW_SIZE, :], y[i: i+WINDOW_SIZE]

            if loc in train_locs:
                train_data.append({'meteo': X_w.astype(np.float32), 'pm25': y_w.astype(np.float32)})
            # elif loc in val_locs:
            #     val_data.append({'meteo': X.astype(np.float32), 'pm25': y.astype(np.float32)})
            elif loc in test_locs:
                test_data.append({'meteo': X_w.astype(np.float32), 'pm25': y_w.astype(np.float32)})
        
    print("---------\t Dataset processing completed \t---------")
    print(f'Time taken: {(time.time()-start_time)/60:.2f} mins')

    return train_data, val_data, test_data

def train(train_loader, test_loader, lr, hidden_size, num_layers):

    model = LSTM(input_size, hidden_size, num_layers, output_size)
    model.to(device)

    # criterion = nn.MSELoss(reduction='none')
    # criterion = RMSE()
    criterion = FrobeniusNorm()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    train_losses, val_losses = [], []
    
    print(f"---------\t Training started lr={lr},  hidden_size={hidden_size}, num_layers={num_layers} \t---------")
    for epoch in range(NUM_EPOCHS):

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels[:, -1, :]

            optimizer.zero_grad()
            preds = model(inputs)

            train_loss = criterion(preds, labels)

            train_loss.backward()
            optimizer.step()

            train_losses.append(train_loss.item())

    
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels[:, -1, :]

                preds = model(inputs)
                val_loss = criterion(preds, labels)

                val_losses.append(val_loss.item())

        if (epoch+1)%2 == 0:
            print(f'Epoch: {epoch+1}/{NUM_EPOCHS}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, \
            time_taken: {(time.time()-start_time)/60:.2f} mins')
        
    print(f"---------\t Training completed lr={lr},  hidden_size={hidden_size}, num_layers={num_layers} \t---------\n")

if __name__ == '__main__':

    # Parser Arguments
    parser = argparse.ArgumentParser()

    # Define command-line arguments
    parser.add_argument('--batch_size', help='Required Batch Size')
    parser.add_argument('--num_epochs', help='Number of Epochs')
    parser.add_argument('--lr', help='Learning Rate')

    # Parse the command-line arguments
    args = parser.parse_args()

    data_file = f'{data_bihar}/bihar_512_sensor_era5_rnn.pkl'
    df = pd.read_pickle(data_file)

    train_locs = load_locs_as_tuples(f'{data_bihar}/train_locations.txt')
    val_locs = load_locs_as_tuples(f'{data_bihar}/val_locations.txt')
    test_locs = load_locs_as_tuples(f'{data_bihar}/test_locations.txt')

    train_locs.extend(val_locs)

    scaler = StandardScaler()
    data = df[[x for x in df.columns if x not in {'timestamp', 'latitude', 'longitude', 'pm25'}]].to_numpy()
    data = scaler.fit_transform(data)
    df[[x for x in df.columns if x not in {'timestamp', 'latitude', 'longitude', 'pm25'}]] = data

    FORECAST_WINDOWS = [6]

    for fw in FORECAST_WINDOWS:

        train_data, _, test_data = data_processing(df, train_locs, val_locs, test_locs, WINDOW_SIZE=100, FORECAST_WINDOW=fw)

        BATCH_SIZE, NUM_EPOCHS, LR = int(args.batch_size), int(args.num_epochs), float(args.lr)
        # print(BATCH_SIZE, NUM_EPOCHS, LR)

        train_dataset = CustomDataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # val_dataset = CustomDataset(val_data)
        # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        test_dataset = CustomDataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        input_size, hidden_size, num_layers, output_size = None, 64, 2, None

        for inputs, labels in train_loader:
            input_size, output_size = inputs.shape[-1], labels.shape[-1]
            input_shape, output_shape = inputs.shape, labels.shape
            break

        print(f'Input Shape: {input_shape} \t Hidden Size: {hidden_size} \t Num Layers: {num_layers} \t Output Size: {output_shape}')
        train(train_loader, test_loader, LR, hidden_size, num_layers)

    # '''
    #     Grid Search begins
    #     Hyperparameters: lr, hidden_size, num_layers
    # '''

    # LR, HIDDEN_SIZE, NUM_LAYERS = [2e-3, 1e-3, 5e-4, 2e-4, 1e-4], [32, 64, 100], [1, 2]

    # for lr in LR:
    #     for hidden_size in HIDDEN_SIZE:
    #         for num_layers in NUM_LAYERS:
    #             train(train_loader, test_loader, lr, hidden_size, num_layers)