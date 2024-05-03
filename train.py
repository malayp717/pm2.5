import numpy as np
from datetime import datetime
import time
import yaml
import os
import sys
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from dataset.TemporalDataset import TemporalDataset
from models.GRU import GRU
from utils import eval_stat

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
config_fp = os.path.join(proj_dir, 'config.yaml')

with open(config_fp, 'r') as f:
    config = yaml.safe_load(f)

# ------------- Config parameters start ------------- #
data_dir = config['filepath']['data_dir']
model_dir = config['filepath']['model_dir']
bihar_pkl_fp = data_dir + config['filepath']['bihar_pkl_fp']
# bihar_npy_fp = data_dir + config['filepath']['bihar_npy_fp']
# locations_fp = data_dir + config['filepath']['locations_fp']
china_npy_fp = data_dir + config['filepath']['china_npy_fp']
china_locations_fp = data_dir + config['filepath']['china_locations_fp']
bihar_map_fp = data_dir + config['filepath']['bihar_map_fp']

batch_size = int(config['train']['batch_size'])
num_epochs = int(config['train']['num_epochs'])
forecast_window = int(config['train']['forecast_window'])
hidden_dim = int(config['train']['hidden_dim'])
lr = float(config['train']['lr'])

update = int(config['dataset']['update'])
data_start = config['dataset']['data_start']
data_end = config['dataset']['data_end']

train_start = config['split']['train_start']
train_end = config['split']['train_end']
val_start = config['split']['val_start']
val_end = config['split']['val_end']
test_start = config['split']['test_start']
test_end = config['split']['test_end']

criterion = nn.MSELoss()
# ------------- Config parameters end   ------------- #

def train(model, loader, optimizer):
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(loader):
        optimizer.zero_grad()
        
        pm25_hist, features, pm25 = data
        pm25 = pm25.to(device)

        pm25_preds = model(features, pm25_hist)

        loss = criterion(pm25, pm25_preds)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= (batch_idx+1)
    return train_loss

def val(model, loader):
    model.eval()
    val_loss = 0

    for batch_idx, data in enumerate(loader):
        
        pm25_hist, features, pm25 = data
        pm25 = pm25.to(device)

        pm25_preds = model(features, pm25_hist)

        loss = criterion(pm25, pm25_preds)
        val_loss += loss.item()
    
    val_loss /= (batch_idx+1)
    return val_loss

def test(model, loader):
    model.eval()
    test_loss = 0

    y, y_pred = [], []

    for batch_idx, data in enumerate(loader):
        
        pm25_hist, features, pm25 = data
        pm25 = pm25.to(device)

        pm25_preds = model(features, pm25_hist)

        loss = criterion(pm25, pm25_preds)
        test_loss += loss.item()

        pm25, pm25_preds = pm25.detach().cpu().numpy(), pm25_preds.detach().cpu().numpy()

        y.extend(pm25)
        y_pred.extend(pm25_preds)
    
    test_loss /= (batch_idx+1)

    y, y_pred = np.array(y), np.array(y_pred)
    print(y.shape, y_pred.shape)
    y, y_pred = y.ravel(), y_pred.ravel()
    print(y.shape, y_pred.shape)

    print(eval_stat(y_pred, y))
    print(y_pred[:48])

    return test_loss

if __name__ == '__main__':
    train_duration = ((datetime(*train_end) - datetime(*train_start)).days + 1) * (24//update)
    val_duration = ((datetime(*val_end) - datetime(*val_start)).days + 1) * (24//update)
    test_duration = ((datetime(*test_end) - datetime(*test_start)).days + 1) * (24//update)

    train_start_index = (datetime(*train_start) - datetime(*data_start)).days * (24//update)
    val_start_index = (datetime(*val_start) - datetime(*data_start)).days * (24//update)
    test_start_index = (datetime(*test_start) - datetime(*data_start)).days * (24//update)

    train_data = TemporalDataset(china_npy_fp, forecast_window, train_start_index, train_duration)
    val_data = TemporalDataset(china_npy_fp, forecast_window, val_start_index, val_duration)
    test_data = TemporalDataset(china_npy_fp, forecast_window, test_start_index, test_duration)

    print(train_duration, val_duration, test_duration, train_duration+val_duration+test_duration)
    print(train_start_index, val_start_index, test_start_index)

    train_data_shape, val_data_shape, test_data_shape = train_data.shape(), val_data.shape(), test_data.shape()

    print(len(train_data), len(val_data), len(test_data))
    print(f'Train Data:\nFeature shape: {train_data_shape[0]} \t PM25_Hist shape: {train_data_shape[1]} \t PM25 shape: {train_data_shape[2]}')
    print(f'Val Data:\nFeature shape: {val_data_shape[0]} \t PM25_Hist shape: {val_data_shape[1]} \t PM25 shape: {val_data_shape[2]}')
    print(f'Test Data:\nFeature shape: {test_data_shape[0]} \t PM25_Hist shape: {test_data_shape[1]} \t PM25 shape: {test_data_shape[2]}')

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    model = GRU(train_data.feature.shape[-1], hidden_dim, 1, train_data.pm25.shape[-1], device)
    model.to(device)
    print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):

        start_time = time.time()

        train_loss = train(model, train_loader, optimizer)
        val_loss = val(model, val_loader)

        if (epoch+1) % 2 == 0:
            print(f'Epoch: {epoch+1}|{num_epochs} \t Train Loss: {train_loss:.4f} \t\
                Val Loss: {val_loss:.4f} \t Time Taken: {(time.time()-start_time)/60:.4f} mins')
    
    train_loss = test(model, train_loader)
    val_loss = test(model, val_loader)
    test_loss = test(model, test_loader)
    print(f'Test Loss: {test_loss:.4f}')