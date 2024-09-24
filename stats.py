import time
import yaml
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Dataset
from models.GRU import GRU
from models.GC_GRU import GC_GRU
from models.GraphConv_GRU import GraphConv_GRU
from models.GNN_GRU import GNN_GRU
from models.Attn_GNN_GRU import Attn_GNN_GRU
from graph import Graph
from china_graph import Graph as cGraph
from utils import eval_stat, load_model

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
config_fp = os.path.join(proj_dir, 'china_config.yaml')

with open(config_fp, 'r') as f:
    config = yaml.safe_load(f)

# ------------- Config parameters start ------------- #
data_dir = config['dirpath']['data_dir']
model_dir = config['dirpath']['model_dir']

npy_fp = data_dir + config['filepath']['npy_fp']
locations_fp = data_dir + config['filepath']['locations_fp']
altitude_fp = data_dir + config['filepath']['altitude_fp']

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
dataset_num = config['dataset']['num']

dist_thresh = float(config['threshold']['distance'])
alt_thresh = float(config['threshold']['altitude'])
haze_thresh = float(config['threshold']['haze'])

train_start = config['split'][dataset_num]['train_start']
train_end = config['split'][dataset_num]['train_end']
val_start = config['split'][dataset_num]['val_start']
val_end = config['split'][dataset_num]['val_end']
test_start = config['split'][dataset_num]['test_start']
test_end = config['split'][dataset_num]['test_end']

criterion = nn.MSELoss()
# ------------- Config parameters end   ------------- #

def get_data_info(hist_len, forecast_len):

    # graph = Graph(locations_fp, dist_thresh)
    graph = cGraph(locations_fp, altitude_fp, dist_thresh, alt_thresh)
    num_locs = graph.num_locs

    train_data = Dataset(npy_fp, forecast_len, hist_len, num_locs, train_start, train_end, data_start, update)
    val_data = Dataset(npy_fp, forecast_len, hist_len, num_locs, val_start, val_end, data_start, update)
    test_data = Dataset(npy_fp, forecast_len, hist_len, num_locs, test_start, test_end, data_start, update)

    return train_data, val_data, test_data, graph

def get_model_info(model_type, train_data, hist_len, forecast_len, graph=None, attn=None):

    assert model_type in {'GRU', 'GC_GRU', 'GraphConv_GRU', 'GNN_GRU', 'Attn_GNN_GRU'},\
                            "Incorrect model type"
    
    if model_type not in {'GRU'}:
        assert graph is not None

    if model_type == 'Attn_GNN_GRU':
        assert attn in {'luong', 'graph'}

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

def test(model, loader, pm25_mean, pm25_std, hist_len):
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

    # hist_len, forecast_len = [24, 48], [12, 24]
    hist_len, forecast_len = [8], [4]
    model_types = ['GRU', 'GC_GRU', 'GraphConv_GRU', 'GNN_GRU', 'Attn_GNN_GRU']
    overall_stats = []

    for hl, fl in zip(hist_len, forecast_len):

        train_data, val_data, test_data, graph = get_data_info(hl, fl)

        train_loader = DataLoader(train_data, drop_last=True, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_data, drop_last=True, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, drop_last=True, batch_size=batch_size, shuffle=False)

        for model_type in model_types:
    
            test_stats = []
            attn = config['train']['attn'] if model_type == 'Attn_GNN_GRU' else None

            for i in range(num_exp):

                model_fp = f'{model_dir}/china_{model_type}_{hl}_{fl}_{i}.pth.tar' if attn == None\
                                    else f'{model_dir}/china_{model_type}_{attn}_{hl}_{fl}_{i}.pth.tar'
                model = get_model_info(model_type, train_data, hl, fl, graph, attn)

                model.to(device)

                curr_epoch, model_state_dict, _, train_losses, val_losses = load_model(model_fp)
                model.load_state_dict(model_state_dict)

                # train_stat = test(model, train_loader, train_data.pm25_mean, train_data.pm25_std)
                # val_stat = test(model, val_loader, train_data.pm25_mean, train_data.pm25_std)
                test_stat = test(model, test_loader, train_data.pm25_mean, train_data.pm25_std, hl)
                test_stats.append(test_stat)

            test_df = pd.DataFrame(data=test_stats)

            model_stats = {col: f'{test_df[col].mean():4f} \u00B1 {test_df[col].std():4f}' for col in test_df.columns}
            model_stats.update({'model': model_type, 'hist_len': hl, 'forecast_len': fl})

            overall_stats.append(model_stats)

    overall_df = pd.DataFrame(data=overall_stats)
    overall_df.to_csv(f'{data_dir}/china_stats.csv', index=False)