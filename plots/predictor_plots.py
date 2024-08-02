import yaml
import os
import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils import load_model
from stats import get_data_info, get_model_info

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_dir)
config_fp = os.path.join(proj_dir, 'config.yaml')

with open(config_fp, 'r') as f:
    config = yaml.safe_load(f)

# ------------- Config parameters start ------------- #
data_dir = config['dirpath']['data_dir']
model_dir = config['dirpath']['model_dir']
plots_dir = config['dirpath']['plots_dir']

npy_fp = data_dir + config['filepath']['npy_fp']
locations_fp = data_dir + config['filepath']['locations_fp']
regions_fp = data_dir + '/bihar_regions.txt'

batch_size = int(config['train']['batch_size'])
num_exp = int(config['train']['num_exp'])
emb_dim = int(config['train']['emb_dim'])
hid_dim = int(config['train']['hid_dim'])
edge_dim = int(config['train']['edge_dim'])

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
# ------------- Config parameters end   ------------- #

region_df = pd.read_csv(regions_fp, delimiter='|')
reg_to_idx = {}
for _, row in region_df.iterrows():
    if row[-1] not in reg_to_idx:
        reg_to_idx[row[-1]] = [row[0]]
    else:
        reg_to_idx[row[-1]].append(row[0])

def test(model, loader, pm25_mean, pm25_std, hist_len, forecast_len):
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
    return y[0::forecast_len, :, :, 0], y_pred[0::forecast_len, :, :, 0]

if __name__ == '__main__':

    hist_len, forecast_len = [24, 48], [12, 24]
    model_type = 'Attn_GNN_GRU'

    for hl, fl in zip(hist_len, forecast_len):

        train_data, val_data, test_data, graph = get_data_info(hl, fl)

        fig, ax = plt.subplots(1, 5, sharey=True, figsize=(20, 4))

        train_loader = DataLoader(train_data, drop_last=True, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_data, drop_last=True, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, drop_last=True, batch_size=batch_size, shuffle=False)

        all_lines, all_labels = [], []

        attn = config['train']['attn'] if model_type == 'Attn_GNN_GRU' else None

        model_fp = f'{model_dir}/{model_type}_{hl}_{fl}_0.pth.tar' if attn == None\
                                    else f'{model_dir}/{model_type}_{attn}_{hl}_{fl}_0.pth.tar'
        model = get_model_info(model_type, train_data, hl, fl, graph, attn)
        model.to(device)

        _, model_state_dict, _, _, _ = load_model(model_fp)
        model.load_state_dict(model_state_dict)

        train_y, train_preds = test(model, train_loader, train_data.pm25_mean, train_data.pm25_std, hl, fl)
        val_y, val_preds = test(model, val_loader, train_data.pm25_mean, train_data.pm25_std, hl, fl)
        test_y, test_preds = test(model, test_loader, train_data.pm25_mean, train_data.pm25_std, hl, fl)

        print(train_y.shape, val_y.shape, test_y.shape)

        for i, (region, locs) in enumerate(reg_to_idx.items()):
            reg_train_y, reg_train_preds = train_y[:, :, locs].transpose(0, 2, 1), train_preds[:, :, locs].transpose(0, 2, 1)
            reg_val_y, reg_val_preds = val_y[:, :, locs].transpose(0, 2, 1), val_preds[:, :, locs].transpose(0, 2, 1)
            reg_test_y, reg_test_preds = test_y[:, :, locs].transpose(0, 2, 1), test_preds[:, :, locs].transpose(0, 2, 1)
            
            reg_train_y, reg_train_preds = np.mean(reg_train_y, axis=1).reshape(-1), np.mean(reg_train_preds, axis=1).reshape(-1)
            reg_val_y, reg_val_preds = np.mean(reg_val_y, axis=1).reshape(-1), np.mean(reg_val_preds, axis=1).reshape(-1) 
            reg_test_y, reg_test_preds = np.mean(reg_test_y, axis=1).reshape(-1), np.mean(reg_test_preds, axis=1).reshape(-1) 

            reg_train_y, reg_train_preds = reg_train_y[::24], reg_train_preds[::24]
            reg_val_y, reg_val_preds = reg_val_y[::24], reg_val_preds[::24]
            reg_test_y, reg_test_preds = reg_test_y[::24], reg_test_preds[::24]

            y, preds = np.concatenate([reg_train_y, reg_val_y, reg_test_y]), np.concatenate([reg_train_preds, reg_val_preds, reg_test_preds])


            xcoords = [len(reg_train_y), len(reg_train_y) + len(reg_val_y)]

            line1, = ax[i].plot(y, label='ground truth')
            line2, = ax[i].plot(preds, label='predictions(AGNN_GRU)')
            ax[i].set_title(region)

            line3 = ax[i].vlines(x=xcoords, ls='--', lw=2, ymin=0, ymax=400, color='black', label='train - val - test')

            if i == 0:
                all_lines.append(line1)
                all_labels.append('ground truth')
                all_lines.append(line2)
                all_labels.append('predictions(AGNN_GRU)')
                line3 = ax[i].vlines(x=xcoords, ls='--', lw=2, ymin=0, ymax=400, color='black', label='train - val - test')
                all_lines.append(line3)
                all_labels.append('train - val - test')
        
        fig.legend(all_lines, all_labels, bbox_to_anchor=(1.0, 1.15), prop={'size': 10})
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(f'{plots_dir}/predictor_plots_{fl}.png', dpi=300, bbox_inches='tight')