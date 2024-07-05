import numpy as np
import time
import yaml
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from dataset import Dataset
from models.GRU import GRU
from models.Seq2Seq_GNN_GRU import Seq2Seq_GNN_GRU
from models.Seq2Seq_Attn_GNN_GRU import Seq2Seq_Attn_GNN_GRU
from models.Seq2Seq_GNN_Transformer import Seq2Seq_GNN_Transformer
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
forecast_len = int(config['train']['forecast_len'])
hist_len = int(config['train']['hist_len'])
hidden_dim = int(config['train']['hidden_dim'])
edge_dim = int(config['train']['edge_dim'])
lr = float(config['train']['lr'])
model_type = config['train']['model']

dataset_num = int(config[location]['dataset']['num'])
update = int(config[location]['dataset']['update'])
data_start = config[location]['dataset']['data_start']
data_end = config[location]['dataset']['data_end']

dist_thresh = float(config[location]['threshold']['distance'])
alt_thresh = float(config[location]['threshold']['altitude']) if location == 'china' else None
haze_thresh = float(config[location]['threshold']['haze'])

train_start = config[location]['split'][dataset_num]['train_start']
train_end = config[location]['split'][dataset_num]['train_end']
val_start = config[location]['split'][dataset_num]['val_start']
val_end = config[location]['split'][dataset_num]['val_end']
test_start = config[location]['split'][dataset_num]['test_start']
test_end = config[location]['split'][dataset_num]['test_end']

criterion = nn.MSELoss()
# ------------- Config parameters end   ------------- #

def get_data_model_info(model_type, location):

    assert location in {'china', 'bihar'}, "Incorrect Location"
    assert model_type in {'GRU', 'GC_GRU', 'Seq2Seq_GC_GRU', 'Seq2Seq_Attn_GC_GRU', 'DGC_GRU', 'Seq2Seq_GNN_GRU',\
                          'Seq2Seq_GNN_Transformer'}, "Incorrect model type"

    graph = Graph(location, locations_fp, dist_thresh, altitude_fp, alt_thresh)
    num_locs = graph.num_locs

    train_data = Dataset(npy_fp, forecast_len, hist_len, num_locs, train_start, train_end, data_start, update)
    val_data = Dataset(npy_fp, forecast_len, hist_len, num_locs, val_start, val_end, data_start, update)
    test_data = Dataset(npy_fp, forecast_len, hist_len, num_locs, test_start, test_end, data_start, update)

    in_dim, city_num = train_data.feature.shape[-1], train_data.feature.shape[-2]
    '''
        Decoder input dim: 3, since the last 3 elements are the only known features during forecasting (is_weekend, cyclic hour embedding)
    '''
    in_dim_dec, num_embeddings = 1, num_locs * (24 // update) * 2
    edge_indices, edge_attr = graph.edge_indices, graph.edge_attr
    u10_mean, u10_std, v10_mean, v10_std = train_data.u10_mean, train_data.u10_std, train_data.v10_mean, train_data.v10_std
    print(edge_indices.size(), edge_attr.size())

    if model_type == 'GRU':
        model = GRU(in_dim, hidden_dim, city_num, hist_len, forecast_len, batch_size, device)
    elif model_type == 'Seq2Seq_GNN_GRU':
        model = Seq2Seq_GNN_GRU(in_dim, in_dim_dec, hidden_dim, city_num, num_embeddings, hist_len, forecast_len,\
                                batch_size, device, edge_indices, edge_attr, u10_mean, u10_std, v10_mean, v10_std, edge_dim)
    elif model_type == 'Seq2Seq_GNN_Transformer':
        model = Seq2Seq_GNN_Transformer(in_dim, in_dim_dec, hidden_dim, city_num, hist_len, forecast_len, batch_size,\
                                        device, edge_indices, edge_attr, u10_mean, u10_std, v10_mean, v10_std)
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

        pm25_label = pm25[:, hist_len:]
        # pm25_hist = pm25[:, :hist_len]

        # pm25_preds = model(features, pm25_hist)
        pm25_preds = model(features, pm25)

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

        pm25_label = pm25[:, hist_len:]
        # pm25_hist = pm25[:, :hist_len]

        # pm25_preds = model(features, pm25_hist)
        pm25_preds = model(features, pm25)

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

        pm25_label = pm25[:, hist_len:]
        # pm25_hist = pm25[:, :hist_len]

        # pm25_preds = model(features, pm25_hist)
        pm25_preds = model(features, pm25)

        loss = criterion(pm25_label, pm25_preds)
        test_loss += loss.item()

        pm25_label = pm25_label * pm25_std + pm25_mean
        pm25_preds = pm25_preds * pm25_std + pm25_mean
        pm25_label, pm25_preds = pm25_label.detach().cpu().numpy(), pm25_preds.detach().cpu().numpy()

        y.extend(pm25_label)
        y_pred.extend(pm25_preds)
    
    test_loss /= (batch_idx+1)

    y, y_pred = np.array(y), np.array(y_pred)
    print(eval_stat(y_pred, y, haze_thresh))
    return test_loss

if __name__ == '__main__':

    train_data, val_data, test_data, model = get_data_model_info(model_type, location)

    # print(f'u10 mean: {train_data.u10_mean} \t u10 std: {train_data.u10_std}')
    # print(f'v10 mean: {train_data.v10_mean} \t v10 std: {train_data.v10_std}')
    # print(f'pm25 mean: {train_data.pm25_mean} \t pm25 std: {train_data.pm25_std}')

    print(f'Train Data:\nFeature shape: {train_data.feature.shape} \t PM25 shape: {train_data.pm25.shape}')
    print(f'Val Data:\nFeature shape: {val_data.feature.shape} \t PM25 shape: {val_data.pm25.shape}')
    print(f'Test Data:\nFeature shape: {test_data.feature.shape} \t PM25 shape: {test_data.pm25.shape}')
    
    train_loader = DataLoader(train_data, drop_last=True, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, drop_last=True, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, drop_last=True, batch_size=batch_size, shuffle=False)

    model.to(device)
    print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    start_time = time.time()
    train_losses, val_losses = [], []
    model_name = model_type + "_" + str(hist_len) + "_" + str(forecast_len) + ".pth.tar"

    for epoch in range(num_epochs):

        train_loss = train(model, train_loader, optimizer)
        val_loss = val(model, val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if (epoch+1) % (num_epochs // 10) == 0:
            print(f'Epoch: {epoch+1}|{num_epochs} \t Train Loss: {train_loss:.4f} \t\
                Val Loss: {val_loss:.4f} \t Time Taken: {(time.time()-start_time)/60:.4f} mins')
            start_time = time.time()

            save_model(model, optimizer, train_losses, val_losses, model_name)
        scheduler.step()
    
    train_loss = test(model, train_loader, train_data.pm25_mean, train_data.pm25_std)
    val_loss = test(model, val_loader, train_data.pm25_mean, train_data.pm25_std)
    test_loss = test(model, test_loader, train_data.pm25_mean, train_data.pm25_std)
    print(f'Test Loss: {test_loss:.4f}')