import time
import yaml
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from dataset import Dataset
from models.GRU import GRU
from models.GC_GRU import GC_GRU
from models.GraphConv_GRU import GraphConv_GRU
from models.GNN_GRU import GNN_GRU
from models.Attn_GNN_GRU import Attn_GNN_GRU
from bihar_graph import Graph as bGraph
from china_graph import Graph as cGraph
from utils import eval_stat, save_model, load_model
from pathlib import Path
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='Choose the Dataset to work on')
args = parser.parse_args()

proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)

config_fp = f'{proj_dir}/{args.config}'
with open(config_fp, 'r') as f:
    config = yaml.safe_load(f)

location = (args.config.split('.')[0]).split('_')[0]

# ------------- Config parameters start ------------- #
data_dir = config['dirpath']['data_dir']
model_dir = config['dirpath']['model_dir']

npy_fp = data_dir + config['filepath']['npy_fp']
locations_fp = data_dir + config['filepath']['locations_fp']
altitude_fp = None if args.config == 'bihar_config.yaml' else data_dir + config['filepath']['altitude_fp']

batch_size = int(config['train']['batch_size'])
num_exp = int(config['train']['num_exp'])
num_epochs = int(config['train']['num_epochs'])
forecast_len = int(config['train']['forecast_len'])
hist_len = int(config['train']['hist_len'])
emb_dim = int(config['train']['emb_dim'])
hid_dim = int(config['train']['hid_dim'])
edge_dim = int(config['train']['edge_dim'])
lr = float(config['train']['lr'])
model_type = config['train']['model']
attn = config['train']['attn'] if model_type == 'Attn_GNN_GRU' else None

update = int(config['dataset']['update'])
data_start = config['dataset']['data_start']
data_end = config['dataset']['data_end']

dist_thresh = float(config['threshold']['distance'])
alt_thresh = None if args.config == 'bihar_config.yaml' else float(config['threshold']['altitude'])
haze_thresh = float(config['threshold']['haze'])

train_start = config['split']['train_start']
train_end = config['split']['train_end']
val_start = config['split']['val_start']
val_end = config['split']['val_end']
test_start = config['split']['test_start']
test_end = config['split']['test_end']

criterion = nn.MSELoss()
# ------------- Config parameters end   ------------- #

def get_data_info():

    graph = bGraph(locations_fp, dist_thresh) if location == 'bihar'\
        else cGraph(locations_fp, altitude_fp, dist_thresh, alt_thresh)
    
    num_locs = graph.num_locs

    train_data = Dataset(npy_fp, forecast_len, hist_len, num_locs, train_start, train_end, data_start, update)
    val_data = Dataset(npy_fp, forecast_len, hist_len, num_locs, val_start, val_end, data_start, update)
    test_data = Dataset(npy_fp, forecast_len, hist_len, num_locs, test_start, test_end, data_start, update)

    return train_data, val_data, test_data, graph

def get_model_info(model_type, train_data, graph=None, attn=None):

    assert model_type in {'GRU', 'GC_GRU', 'GraphConv_GRU', 'GNN_GRU', 'Attn_GNN_GRU'},\
                            "Incorrect model type"
    
    if model_type not in {'GRU'}:
        assert graph is not None

    if model_type == 'Attn_GNN_GRU':
        assert attn == 'luong'

    in_dim, city_num = train_data.feature.shape[-1], train_data.feature.shape[-2]
    num_locs = graph.num_locs
    '''
        Decoder input dim: 3, since the last 3 elements are the only known features during forecasting (is_weekend, cyclic hour embedding)
    '''
    in_dim_dec, num_embeddings = 1, num_locs * (24 // update) * 2
    edge_indices, edge_weights, edge_attr = graph.edge_indices, graph.edge_weights, graph.edge_attr
    u10_mean, u10_std, v10_mean, v10_std = train_data.u10_mean, train_data.u10_std, train_data.v10_mean, train_data.v10_std
    # print(f'Edge Indices Shape: {edge_indices.size()}, Edge Attr Shape: {edge_attr.size()}')

    if model_type == 'GRU':
        model = GRU(in_dim, in_dim_dec, emb_dim, hid_dim, city_num, num_embeddings, hist_len, forecast_len, batch_size, device)
    elif model_type == 'GC_GRU':
        model = GC_GRU(in_dim, in_dim_dec, emb_dim, hid_dim, city_num, num_embeddings, hist_len, forecast_len,\
                               batch_size, device, edge_indices)
    elif model_type == 'GraphConv_GRU':
        model = GraphConv_GRU(in_dim, in_dim_dec, emb_dim, hid_dim, city_num, num_embeddings, hist_len, forecast_len,\
                               batch_size, device, edge_indices, edge_weights)
    elif model_type == 'GNN_GRU':
        model = GNN_GRU(in_dim, in_dim_dec, emb_dim, hid_dim, city_num, num_embeddings, hist_len, forecast_len,\
                                batch_size, device, edge_indices, edge_attr, u10_mean, u10_std, v10_mean, v10_std, edge_dim)
    elif model_type == 'Attn_GNN_GRU':
        model = Attn_GNN_GRU(in_dim, in_dim_dec, emb_dim, hid_dim, city_num, num_embeddings, hist_len, forecast_len,\
                                batch_size, device, edge_indices, edge_attr, u10_mean, u10_std, v10_mean, v10_std, edge_dim, attn)
    else:
        raise Exception('Wrong model name!')

    return model

def train(model, loader, optimizer):
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(loader):
        optimizer.zero_grad()
        
        features, pm25 = data
        pm25 = pm25.to(device)

        pm25_label = pm25[:, hist_len:]
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
    stats_dict = eval_stat(y_pred, y, haze_thresh)
    stats_dict.update({'loss': round(test_loss, 4)})
    return stats_dict

if __name__ == '__main__':
    train_data, val_data, test_data, graph = get_data_info()
    
    train_loader = DataLoader(train_data, drop_last=True, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, drop_last=True, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, drop_last=True, batch_size=batch_size, shuffle=False)

    train_stats, val_stats, test_stats = [], [], []

    for i in range(num_exp):
        print(f'----------------------- Experiment number: {i} start -----------------------')

        model = get_model_info(model_type, train_data, graph, attn)

        model.to(device)
        print(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-3)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

        start_time = time.time()
        model_fp = f'{model_dir}/{location}_{model_type}_{hist_len}_{forecast_len}_{i}.pth.tar' if attn == None\
                    else f'{model_dir}/{location}_{model_type}_{attn}_{hist_len}_{forecast_len}_{i}.pth.tar'

        if Path(model_fp).is_file():
            model_state_dict, optimizer_state_dict = load_model(model_fp)
            model.load_state_dict(model_state_dict)
            optimizer.load_state_dict(optimizer_state_dict)

        for epoch in range(0, num_epochs):

            train_loss = train(model, train_loader, optimizer)
            val_loss = val(model, val_loader)

            if (epoch+1)%(num_epochs//10) == 0:
                print(f'Epoch: {epoch+1}|{num_epochs} \t Train Loss: {train_loss:.4f} \t\
                    Val Loss: {val_loss:.4f} \t Time Taken: {(time.time()-start_time)/60:.4f} mins')
                
                start_time = time.time()

            save_model(model, optimizer, model_fp)
            scheduler.step()

        
        train_stat = test(model, train_loader, train_data.pm25_mean, train_data.pm25_std)
        val_stat = test(model, val_loader, train_data.pm25_mean, train_data.pm25_std)
        test_stat = test(model, test_loader, train_data.pm25_mean, train_data.pm25_std)

        train_stats.append(train_stat)
        val_stats.append(val_stat)
        test_stats.append(test_stat)

        print(f'Train: {train_stat}')
        print(f'Val: {val_stat}')
        print(f'Test: {test_stat}')
        print(f'----------------------- Experiment number: {i} end -----------------------\n\n')