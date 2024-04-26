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
epochs = int(config['train']['epochs'])
forecast_window = int(config['train']['forecast_window'])
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

    train_duration = ((datetime(*train_end) - datetime(*train_start)).days + 1) * (24//update)
    val_duration = ((datetime(*val_end) - datetime(*val_start)).days + 1) * (24//update)
    test_duration = ((datetime(*test_end) - datetime(*test_start)).days + 1) * (24//update)

    train_start_index = (datetime(*train_start) - datetime(*data_start)).days * (24//update)
    val_start_index = (datetime(*val_start) - datetime(*data_start)).days * (24//update)
    test_start_index = (datetime(*test_start) - datetime(*data_start)).days * (24//update)

    train_data = TemporalDataset(bihar_npy_fp, forecast_window, train_start_index, train_duration)
    val_data = TemporalDataset(bihar_npy_fp, forecast_window, val_start_index, val_duration)
    test_data = TemporalDataset(bihar_npy_fp, forecast_window, test_start_index, test_duration)

    print(train_duration, val_duration, test_duration, train_duration+val_duration+test_duration)
    print(train_start_index, val_start_index, test_start_index)

    print(len(train_data), len(val_data), len(test_data))
    print(train_data.shape())
    print(val_data.shape())
    print(test_data.shape())