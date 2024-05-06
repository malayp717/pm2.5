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
from dataset.SpatioDataset import SpatioTemporalDataset
from models.GRU import GRU
from models.GC_GRU import GC_GRU
from graph import Graph
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
bihar_npy_fp = data_dir + config['filepath']['bihar_npy_fp']
bihar_locations_fp = data_dir + config['filepath']['bihar_locations_fp']
china_npy_fp = data_dir + config['filepath']['china_npy_fp']
china_locations_fp = data_dir + config['filepath']['china_locations_fp']
bihar_map_fp = data_dir + config['filepath']['bihar_map_fp']

batch_size = int(config['train']['batch_size'])
num_epochs = int(config['train']['num_epochs'])
forecast_window = int(config['train']['forecast_window'])
hist_window = int(config['train']['hist_window'])
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
        
        features, pm25 = data
        pm25 = pm25.to(device)

        pm25_label = pm25[:, hist_window:]
        pm25_hist = pm25[:, :hist_window]

        pm25_preds = model(features, pm25_hist)

        loss = criterion(pm25_label, pm25_preds)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= (batch_idx+1)
    return train_loss

def val(model, loader):
    model.eval()
    val_loss = 0

    for batch_idx, data in enumerate(loader):
        
        features, pm25 = data
        pm25 = pm25.to(device)

        pm25_label = pm25[:, hist_window:]
        pm25_hist = pm25[:, :hist_window]

        pm25_preds = model(features, pm25_hist)

        loss = criterion(pm25_label, pm25_preds)
        val_loss += loss.item()
    
    val_loss /= (batch_idx+1)
    return val_loss

def test(model, loader):
    model.eval()
    test_loss = 0

    y, y_pred = [], []

    for batch_idx, data in enumerate(loader):
        
        features, pm25 = data
        pm25 = pm25.to(device)

        pm25_label = pm25[:, hist_window:]
        pm25_hist = pm25[:, :hist_window]

        pm25_preds = model(features, pm25_hist)

        loss = criterion(pm25_label, pm25_preds)
        test_loss += loss.item()

        pm25_label, pm25_preds = pm25_label.detach().cpu().numpy(), pm25_preds.detach().cpu().numpy()

        y.extend(pm25_label)
        y_pred.extend(pm25_preds)
    
    test_loss /= (batch_idx+1)

    y, y_pred = np.array(y), np.array(y_pred)
    y, y_pred = y.ravel(), y_pred.ravel()

    print(eval_stat(y_pred, y))
    return test_loss

if __name__ == '__main__':
    # train_data = TemporalDataset(bihar_npy_fp, forecast_window, hist_window, train_start, train_end, data_start, update)
    # val_data = TemporalDataset(bihar_npy_fp, forecast_window, hist_window, val_start, val_end, data_start, update)
    # test_data = TemporalDataset(bihar_npy_fp, forecast_window, hist_window, test_start, test_end, data_start, update)

    graph = Graph(bihar_locations_fp)

    train_data = SpatioTemporalDataset(bihar_npy_fp, forecast_window, hist_window, train_start, train_end, data_start, update, graph.edge_indices)
    val_data = SpatioTemporalDataset(bihar_npy_fp, forecast_window, hist_window, val_start, val_end, data_start, update, graph.edge_indices)
    test_data = SpatioTemporalDataset(bihar_npy_fp, forecast_window, hist_window, test_start, test_end, data_start, update, graph.edge_indices)

    print(len(train_data), len(val_data), len(test_data))
    print(f'Train Data:\nFeature shape: {train_data.feature.shape} \t PM25 shape: {train_data.pm25.shape}')
    print(f'Val Data:\nFeature shape: {val_data.feature.shape} \t PM25 shape: {val_data.pm25.shape}')
    print(f'Test Data:\nFeature shape: {test_data.feature.shape} \t PM25 shape: {test_data.pm25.shape}')
    
    train_loader = torch.utils.data.DataLoader(train_data, drop_last=True, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_data, drop_last=True, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, drop_last=True, batch_size=batch_size)

    in_dim, out_dim = train_data.feature.shape[-1], train_data.pm25.shape[-1]
    city_num = train_data.feature.shape[-2]

    # model = GRU(in_dim, hidden_dim, out_dim, city_num, hist_window, forecast_window, batch_size, device)
    model = GC_GRU(in_dim, hidden_dim, out_dim, city_num, hist_window, forecast_window, batch_size, device, graph.edge_indices)
    model.to(device)
    print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    start_time = time.time()
    for epoch in range(num_epochs):

        train_loss = train(model, train_loader, optimizer)
        val_loss = val(model, val_loader)

        if (epoch+1) % 2 == 0:
            print(f'Epoch: {epoch+1}|{num_epochs} \t Train Loss: {train_loss:.4f} \t\
                Val Loss: {val_loss:.4f} \t Time Taken: {(time.time()-start_time)/60:.4f} mins')
            start_time = time.time()
    
    train_loss = test(model, train_loader)
    val_loss = test(model, val_loader)
    test_loss = test(model, test_loader)
    print(f'Test Loss: {test_loss:.4f}')