import numpy as np
import time
import yaml
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.optim import lr_scheduler
from Dataset import Dataset
from models.GRU import GRU
from models.GC_GRU import GC_GRU
# from models.DGC_GRU import DGC_GRU
from models.Seq2Seq_GC_GRU import Seq2Seq_GC_GRU
from models.Seq2Seq_Attn_GC_GRU import Seq2Seq_Attn_GC_GRU
from graph import Graph
from utils import eval_stat, save_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
config_fp = os.path.join(proj_dir, 'config.yaml')

with open(config_fp, 'r') as f:
    config = yaml.safe_load(f)

# ------------- Config parameters start ------------- #
data_dir = config['dirpath']['data_dir']
model_dir = config['dirpath']['model_dir']

location = config['location']

npy_fp = data_dir + config[location]['filepath']['npy_fp']
locations_fp = data_dir + config[location]['filepath']['locations_fp']
altitude_fp = data_dir + config[location]['filepath']['altitude_fp'] if location == 'china' else None
# map_fp = data_dir + config[location]['filepath']['map_fp'] if location == 'bihar' else None

batch_size = int(config['train']['batch_size'])
num_epochs = int(config['train']['num_epochs'])
forecast_window = int(config['train']['forecast_window'])
hist_window = int(config['train']['hist_window'])
hidden_dim = int(config['train']['hidden_dim'])
lr = float(config['train']['lr'])
model_type = config['train']['model']

update = int(config[location]['dataset']['update'])
data_start = config[location]['dataset']['data_start']
data_end = config[location]['dataset']['data_end']

dist_thresh = float(config[location]['threshold']['distance'])
alt_thresh = float(config[location]['threshold']['altitude']) if location == 'china' else None
haze_thresh = float(config[location]['threshold']['haze'])

train_start = config[location]['split']['train_start']
train_end = config[location]['split']['train_end']
val_start = config[location]['split']['val_start']
val_end = config[location]['split']['val_end']
test_start = config[location]['split']['test_start']
test_end = config[location]['split']['test_end']

criterion = nn.MSELoss()
# ------------- Config parameters end   ------------- #

def get_data_model_info(model_type, location):

    assert location in {'china', 'bihar'}, "Incorrect Location"
    assert model_type in {'GRU', 'GC_GRU', 'Seq2Seq_GC_GRU', 'Seq2Seq_Attn_GC_GRU', 'DGC_GRU'}, "Incorrect model type"

    train_data = Dataset(npy_fp, forecast_window, hist_window, train_start, train_end, data_start, update)
    val_data = Dataset(npy_fp, forecast_window, hist_window, train_start, train_end, data_start, update)
    test_data = Dataset(npy_fp, forecast_window, hist_window, train_start, train_end, data_start, update)

    graph = Graph(location, locations_fp, dist_thresh, altitude_fp, alt_thresh)

    in_dim, city_num = train_data.feature.shape[-1], train_data.feature.shape[-2]

    if model_type == 'GRU':
        model = GRU(in_dim, hidden_dim, city_num, hist_window, forecast_window, batch_size, device)
    elif model_type == 'GC_GRU':
        model = GC_GRU(in_dim, hidden_dim, city_num, hist_window, forecast_window, batch_size, device, graph.adj_mat)
    elif model_type == 'Seq2Seq_GC_GRU':
        model = Seq2Seq_GC_GRU(in_dim, hidden_dim, city_num, hist_window, forecast_window, batch_size, device, graph.adj_mat)
    elif model_type == 'Seq2Seq_Attn_GC_GRU':
        model = Seq2Seq_Attn_GC_GRU(in_dim, hidden_dim, city_num, hist_window, forecast_window, batch_size, device, graph.adj_mat)
    # elif model_type == 'DGC_GRU':
    #     model = DGC_GRU(in_dim, hidden_dim, city_num, hist_window, forecast_window, batch_size, device, graph.adj_mat, graph.angles)
    else:
        raise Exception('Wrong model name!')

    return train_data, val_data, test_data, model

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

def test(model, loader, pm25_mean, pm25_std):
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

        pm25_label = pm25_label * pm25_std + pm25_mean
        pm25_preds = pm25_preds * pm25_std + pm25_mean
        pm25_label, pm25_preds = pm25_label.detach().cpu().numpy(), pm25_preds.detach().cpu().numpy()

        y.extend(pm25_label)
        y_pred.extend(pm25_preds)
    
    test_loss /= (batch_idx+1)

    y, y_pred = np.array(y), np.array(y_pred)
    y, y_pred = y.ravel(), y_pred.ravel()

    print(eval_stat(y_pred, y, haze_thresh))
    return test_loss

if __name__ == '__main__':

    train_data, val_data, test_data, model = get_data_model_info(model_type, location)
    pm25_mean, pm25_std = train_data.pm25_mean, train_data.pm25_std

    print(f'Train Data:\nFeature shape: {train_data.feature.shape} \t PM25 shape: {train_data.pm25.shape}')
    print(f'Val Data:\nFeature shape: {val_data.feature.shape} \t PM25 shape: {val_data.pm25.shape}')
    print(f'Test Data:\nFeature shape: {test_data.feature.shape} \t PM25 shape: {test_data.pm25.shape}')
    
    train_loader = DataLoader(train_data, drop_last=True, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, drop_last=True, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, drop_last=True, batch_size=batch_size, shuffle=False)

    model.to(device)
    print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    start_time = time.time()
    train_losses, val_losses = [], []
    model_name = model_type + "_" + str(hist_window) + "_" + str(forecast_window) + ".pth.tar"

    for epoch in range(num_epochs):

        train_loss = train(model, train_loader, optimizer)
        val_loss = val(model, val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if (epoch+1) % (num_epochs // 10) == 0:
            print(f'Epoch: {epoch+1}|{num_epochs} \t Train Loss: {train_loss:.4f} \t\
                Val Loss: {val_loss:.4f} \t Time Taken: {(time.time()-start_time)/60:.4f} mins')
            start_time = time.time()

            # save_model(model, optimizer, train_losses, val_losses, model_name)
    
    train_loss = test(model, train_loader, pm25_mean, pm25_std)
    val_loss = test(model, val_loader, pm25_mean, pm25_std)
    test_loss = test(model, test_loader, pm25_mean, pm25_std)
    print(f'Test Loss: {test_loss:.4f}')