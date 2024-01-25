import sys
import os
sys.path.append('..')
import time
import pickle
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
from sklearn.ensemble import RandomTreesEmbedding, RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
import re
from utils import *
from constants import *
from rnn import RNN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __name__ == '__main__':
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(device)

    cols = ['timestamp', 'latitude', 'longitude', 'rh', 'temp', 'blh', 'u10', 'v10', 'kx', 'sp', 'tp', 'pm25']
    data_file = f'{data_bihar}/bihar_512_sensor_era5_image_uts.pkl'
    df = pd.read_pickle(data_file)

    df = df[cols]
    print(df.dtypes)
    print(df.shape)
    print(df.count())