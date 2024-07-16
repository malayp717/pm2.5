import yaml
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Dataset
from models.Attn_GNN_GRU import Attn_GNN_GRU
from graph import Graph
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
config_fp = os.path.join(proj_dir, 'config.yaml')

with open(config_fp, 'r') as f:
    config = yaml.safe_load(f)

# ------------- Config parameters start ------------- #
data_dir = config['dirpath']['data_dir']
model_dir = config['dirpath']['model_dir']

npy_fp = data_dir + config['filepath']['npy_fp']
locations_fp = data_dir + config['filepath']['locations_fp']

batch_size = int(config['train']['batch_size'])
forecast_len = int(config['train']['forecast_len'])
hist_len = int(config['train']['hist_len'])
# emb_dim = int(config['train']['emb_dim'])
# hid_dim = int(config['train']['hid_dim'])
edge_dim = int(config['train']['edge_dim'])
# lr = float(config['train']['lr'])
model_type = config['train']['model']
attn = config['train']['attn'] if model_type == 'Attn_GNN_GRU' else None

update = int(config['dataset']['update'])
data_start = config['dataset']['data_start']
data_end = config['dataset']['data_end']

dist_thresh = float(config['threshold']['distance'])
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

    graph = Graph(locations_fp, dist_thresh)
    num_locs = graph.num_locs

    train_data = Dataset(npy_fp, forecast_len, hist_len, num_locs, train_start, train_end, data_start, update)
    val_data = Dataset(npy_fp, forecast_len, hist_len, num_locs, val_start, val_end, data_start, update)
    test_data = Dataset(npy_fp, forecast_len, hist_len, num_locs, test_start, test_end, data_start, update)

    return train_data, val_data, test_data, graph

def get_model_info(model_type, train_data, graph, attn, hid_dim, emb_dim):

    assert model_type == 'Attn_GNN_GRU', "Incorrect model type"

    in_dim, city_num = train_data.feature.shape[-1], train_data.feature.shape[-2]
    num_locs = graph.num_locs
    '''
        Decoder input dim: 3, since the last 3 elements are the only known features during forecasting (is_weekend, cyclic hour embedding)
    '''
    in_dim_dec, num_embeddings = 1, num_locs * (24 // update) * 2
    edge_indices, edge_attr = graph.edge_indices, graph.edge_attr
    u10_mean, u10_std, v10_mean, v10_std = train_data.u10_mean, train_data.u10_std, train_data.v10_mean, train_data.v10_std
    # print(f'Edge Indices Shape: {edge_indices.size()}, Edge Attr Shape: {edge_attr.size()}')

    model = Attn_GNN_GRU(in_dim, in_dim_dec, emb_dim, hid_dim, city_num, num_embeddings, hist_len, forecast_len,\
                                batch_size, device, edge_indices, edge_attr, u10_mean, u10_std, v10_mean, v10_std, edge_dim, attn)

    return model

def train_epoch(model, loader, optimizer):
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

def val_epoch(model, loader):
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

def train(config=None):

    with wandb.init(config=config):
        config = wandb.config

        train_data, val_data, _, graph = get_data_info()
        train_loader = DataLoader(train_data, drop_last=True, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_data, drop_last=True, batch_size=batch_size, shuffle=False)

        print(f'SWEEP PARAMS:\nNUM_EPOCHS: {config.num_epochs}\nHID_DIM: {config.hid_dim}\nEMB_DIM: {config.emb_dim}\nLR: {config.lr}\
                \nWEIGHT_DECAY: {config.weight_decay}')

        model = get_model_info(model_type, train_data, graph, attn, config.hid_dim, config.emb_dim)

        model.to(device)
        print(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        for epoch in range(config.num_epochs):

            _ = train_epoch(model, train_loader, optimizer)
            val_loss = val_epoch(model, val_loader)
            
            wandb.log({'val_loss': val_loss, 'epoch': epoch})


if __name__ == '__main__':
    sweep_id = 'qrdm02rw'
    wandb.login()
    wandb.agent(sweep_id, train, count=20)