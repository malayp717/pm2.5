import os
import sys
import yaml
import numpy as np
from datetime import datetime
from dataset.TemporalDataset import TemporalDataset

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
locations_fp = data_dir + config['filepath']['locations_fp']
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
# ------------- Config parameters end   ------------- #

if __name__ == '__main__':

    train_data = TemporalDataset(bihar_npy_fp, forecast_window, hist_window, train_start, train_end, data_start, update)
    val_data = TemporalDataset(bihar_npy_fp, forecast_window, hist_window, val_start, val_end, data_start, update)
    test_data = TemporalDataset(bihar_npy_fp, forecast_window, hist_window, test_start, test_end, data_start, update)

    print(len(train_data), len(val_data), len(test_data))
    print(f'Train Data:\nFeature shape: {train_data.feature.shape} \t PM25 shape: {train_data.pm25.shape}')
    print(f'Val Data:\nFeature shape: {val_data.feature.shape} \t PM25 shape: {val_data.pm25.shape}')
    print(f'Test Data:\nFeature shape: {test_data.feature.shape} \t PM25 shape: {test_data.pm25.shape}')