import time
import numpy as np
import time
from pathlib import Path
import math
from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
import pandas as pd
from utils import *
from constants import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from models.LSTM import LSTM, CustomDataset, FrobeniusNorm

import argparse
import warnings
warnings.filterwarnings('ignore')

def train(train_loader, val_loader, test_loader, lr, input_size, output_size, hidden_size, num_layers, FW):

    model = LSTM(input_size, hidden_size, num_layers, output_size, bidirectional=True)
    model.to(device)

    criterion = FrobeniusNorm()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    start_time = time.time()
    start_epoch, train_losses, val_losses = 0, [], []

    model_path = f'{model_dir}/BLSTM_{FW}.pth.tar'
    model_file = Path(model_path)

    if model_file.is_file():
        state = torch.load(model_path)
        model.load_state_dict(state['state'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch, train_losses, val_losses = state['epoch'], state['train_losses'], state['val_losses']
    
    print(f"---------\t Training started lr={lr},  hidden_size={hidden_size}, num_layers={num_layers} \t---------")
    for epoch in range(start_epoch, NUM_EPOCHS):

        loss_train, loss_val = [], []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels[:, -1, :]

            optimizer.zero_grad()
            preds = model(inputs)

            train_loss = criterion(preds, labels)

            train_loss.backward()
            optimizer.step()

            loss_train.append(train_loss.item())
    
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels[:, -1, :]

                preds = model(inputs)
                val_loss = criterion(preds, labels)

                loss_val.append(val_loss.item())

        train_loss, val_loss = np.mean(loss_train), np.mean(loss_val)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        state = {
             'epoch': epoch,
             'state': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'train_losses': train_losses,
             'val_losses': val_losses
        }

        torch.save(state, f'{model_dir}/BLSTM_{FW}.pth.tar')

        # if (epoch+1)%2 == 0:
        print(f'Epoch: {epoch+1}/{NUM_EPOCHS}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, \
            time_taken: {(time.time()-start_time)/60:.2f} mins')
        
    print(f"---------\t Training completed lr={lr},  hidden_size={hidden_size}, num_layers={num_layers} \t---------")


def get_stats(train_loader, val_loader, test_loader, input_size, output_size, hidden_size, num_layers, FW):

    model = LSTM(input_size, hidden_size, num_layers, output_size, bidirectional=True)
    model.to(device)

    start_time = time.time()
    # start_epoch, train_losses, val_losses = 0, [], []

    model_path = f'{model_dir}/BLSTM_{FW}.pth.tar'
    model_file = Path(model_path)

    if model_file.is_file():
        state = torch.load(model_path)
        model.load_state_dict(state['state'])
        # start_epoch, train_losses, val_losses = state['epoch'], state['train_losses'], state['val_losses']
    else:
        print("No pth file found")

    print(f"---------\t Stats \t---------")
    start_time = time.time()
    y_train, y_train_pred, y_val, y_val_pred, y_test, y_test_pred = [], [], [], [], [], []

    with torch.no_grad():
        for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels[:, -1, :]

                preds = model(inputs)
                preds = torch.clamp(preds, LOWER_BOUND, UPPER_BOUND)

                y_train.extend(labels.cpu().tolist())
                y_train_pred.extend(preds.cpu().tolist())

        for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels[:, -1, :]

                preds = model(inputs)
                preds = torch.clamp(preds, LOWER_BOUND, UPPER_BOUND)

                y_val.extend(labels.cpu().tolist())
                y_val_pred.extend(preds.cpu().tolist())

        for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels[:, -1, :]

                preds = model(inputs)
                preds = torch.clamp(preds, LOWER_BOUND, UPPER_BOUND)

                y_test.extend(labels.cpu().tolist())
                y_test_pred.extend(preds.cpu().tolist())

    y_train, y_train_pred, y_val, y_val_pred, y_test, y_test_pred = np.array(y_train), np.array(y_train_pred), np.array(y_val),\
                                                                np.array(y_val_pred), np.array(y_test), np.array(y_test_pred)
    
    y_train, y_train_pred, y_val, y_val_pred, y_test, y_test_pred = y_train.reshape(-1), y_train_pred.reshape(-1), y_val.reshape(-1),\
        y_val_pred.reshape(-1), y_test.reshape(-1), y_test_pred.reshape(-1)
    
    # print(y_train.shape, y_train_pred.shape, y_val.shape, y_val_pred.shape, y_test.shape, y_test_pred.shape)

    print(f"Train Stats (RMSE, R_squared, p_value, R_squared_pearson, p_value_pearson)")
    print(eval_stat(y_train, y_train_pred))
    print(f"Val Stats (RMSE, R_squared, p_value, R_squared_pearson, p_value_pearson)")
    print(eval_stat(y_val, y_val_pred))
    print(f"Test Stats (RMSE, R_squared, p_value, R_squared_pearson, p_value_pearson)")
    print(eval_stat(y_test, y_test_pred))

    print(f"---------\t Stats Completed\t Time Taken={(time.time()-start_time)/60:.2f} mins\t---------\n")


if __name__ == '__main__':

    # Parser Arguments
    parser = argparse.ArgumentParser()

    # Define command-line arguments
    parser.add_argument('--batch_size', help='Required Batch Size')
    parser.add_argument('--num_epochs', help='Number of Epochs')
    parser.add_argument('--lr', help='Learning Rate')
    # parser.add_argument('--forecast_window', nargs='+', type=int, help='List of forecast window values')

    # Parse the command-line arguments
    args = parser.parse_args()

    data_file = f'{data_bihar}/bihar_512_sensor_era5_rnn.pkl'
    df = pd.read_pickle(data_file)

    train_locs = load_locs_as_tuples(f'{data_bihar}/train_locations.txt')
    val_locs = load_locs_as_tuples(f'{data_bihar}/val_locations.txt')
    test_locs = load_locs_as_tuples(f'{data_bihar}/test_locations.txt')

    scaler = StandardScaler()
    data = df[[x for x in df.columns if x not in {'timestamp', 'latitude', 'longitude', 'pm25'}]].to_numpy()
    data = scaler.fit_transform(data)
    df[[x for x in df.columns if x not in {'timestamp', 'latitude', 'longitude', 'pm25'}]] = data

    FORECAST_WINDOWS = [1, 6, 12, 24]

    for FW in FORECAST_WINDOWS:

        train_data, val_data, test_data = data_processing(df, train_locs, val_locs, test_locs, WS=168, FW=FW)

        # print(len(train_data), len(val_data), len(test_data))
        # print(train_data[0]['pm25'].shape)

        BATCH_SIZE, NUM_EPOCHS, LR = int(args.batch_size), int(args.num_epochs), float(args.lr)
        # print(BATCH_SIZE, NUM_EPOCHS, LR)

        train_dataset = CustomDataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

        val_dataset = CustomDataset(val_data)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        test_dataset = CustomDataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        input_size, hidden_size, num_layers, output_size = None, 64, 2, None

        for inputs, labels in train_loader:
            input_size, output_size = inputs.shape[-1], labels.shape[-1]
            input_shape, output_shape = inputs.shape, labels.shape
            break

        print(f'Input Shape: {input_shape} \t Hidden Size: {hidden_size} \t Num Layers: {num_layers} \t Output Size: {output_shape}')
        train(train_loader, val_loader, test_loader, LR, input_size, output_size, hidden_size, num_layers, FW)
        get_stats(train_loader, val_loader, test_loader, input_size, output_size, hidden_size, num_layers, FW)

    # '''
    #     Grid Search begins
    #     Hyperparameters: lr, hidden_size, num_layers
    # '''

    # LR, HIDDEN_SIZE, NUM_LAYERS = [2e-3, 1e-3, 5e-4, 2e-4, 1e-4], [32, 64, 100], [1, 2]

    # for lr in LR:
    #     for hidden_size in HIDDEN_SIZE:
    #         for num_layers in NUM_LAYERS:
    #             train(train_loader, test_loader, lr, hidden_size, num_layers)