import sys
sys.path.append('../')
import os
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from train import get_data_info, get_model_info
from utils import load_model
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(f'{proj_dir}/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# ------------- Config parameters start ------------- #
data_dir = config['dirpath']['data_dir']
model_dir = config['dirpath']['model_dir']
plots_dir = config['dirpath']['plots_dir']

location = config['location']
# map_fp = data_dir + config[location]['filepath']['map_fp'] if location == 'bihar' else None

batch_size = int(config['train']['batch_size'])
forecast_len = int(config['train']['forecast_len'])
hist_len = int(config['train']['hist_len'])
model_type = config['train']['model']
attn = config['train']['attn'] if model_type == 'Attn_GNN_GRU' else None
# ------------- Config parameters end   ------------- #

def test(model, loader, pm25_mean, pm25_std, single_day):
    model.eval()
    y, y_pred = [], []

    for _, data in enumerate(loader):
        
        features, pm25 = data
        pm25 = pm25.to(device)

        pm25_label = pm25[:, hist_len:]
        pm25_preds = model(features, pm25)

        pm25_label = pm25_label * pm25_std + pm25_mean
        pm25_preds = pm25_preds * pm25_std + pm25_mean
        pm25_label, pm25_preds = pm25_label.detach().cpu().numpy(), pm25_preds.detach().cpu().numpy()

        y.extend(pm25_label)
        y_pred.extend(pm25_preds)

    y, y_pred = np.array(y), np.array(y_pred)
    if single_day:
        return y[0, :, 0, 0].reshape(-1), y_pred[0, :, 0, 0].reshape(-1)
    return y[0::forecast_len, :, 0, 0].T.reshape(-1), y_pred[0::forecast_len, :, 0, 0].T.reshape(-1)

if __name__ == '__main__':

    train_data, val_data, test_data, graph = get_data_info(location)
    single_day = False

    train_loader = DataLoader(train_data, drop_last=True, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, drop_last=True, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, drop_last=True, batch_size=batch_size, shuffle=False)

    # print(f'Train Data:\nFeature shape: {train_data.feature.shape} \t PM25 shape: {train_data.pm25.shape}')
    # print(f'Val Data:\nFeature shape: {val_data.feature.shape} \t PM25 shape: {val_data.pm25.shape}')
    # print(f'Test Data:\nFeature shape: {test_data.feature.shape} \t PM25 shape: {test_data.pm25.shape}')

    model = get_model_info(model_type, train_data, graph, attn)
    model.to(device)

    model_fp = f'{model_dir}/{model_type}_{hist_len}_{forecast_len}_0.pth.tar' if attn == None\
                    else f'{model_dir}/{model_type}_{attn}_{hist_len}_{forecast_len}_0.pth.tar'

    if Path(model_fp).is_file():
            curr_epoch, model_state_dict, _, train_losses, val_losses = load_model(model_fp)
            model.load_state_dict(model_state_dict)

    train_y, train_y_pred = test(model, train_loader, train_data.pm25_mean, train_data.pm25_std, single_day)
    val_y, val_y_pred = test(model, val_loader, train_data.pm25_mean, train_data.pm25_std, single_day)
    test_y, test_y_pred = test(model, test_loader, train_data.pm25_mean, train_data.pm25_std, single_day)

    y, y_pred = np.concatenate([train_y, val_y, test_y]), np.concatenate([train_y_pred, val_y_pred, test_y_pred])
    print(train_y.shape, val_y.shape, test_y.shape)
    print(y.shape, y_pred.shape)

    xcoords = [x.shape[0] for x in [train_y, val_y, test_y]]
    xcoords[1] += xcoords[0]
    xcoords[2] += xcoords[1]

    plt.figure(figsize=(10, 6))
    plt.plot(y, label='true label')
    plt.plot(y_pred, label='preds')
    plt.vlines(x=xcoords, ls='--', lw=2, ymin=0, ymax=max(y.max(), y_pred.max()), color='black', label='train-val-test')
    plt.legend(bbox_to_anchor=(1.0, 1.15), prop={'size': 10})
    plt.savefig(f'{plots_dir}/{model_type}_fit_{single_day}.jpg', dpi=200)