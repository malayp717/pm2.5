import sys
sys.path.append('..')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time
from pathlib import Path
from constants import *
import pickle
import argparse

def save_data(in_file, monthwise_split, FW, WS):

    df = pd.read_pickle(in_file)
    df = df[[x for x in df.columns if x not in {'block', 'district'}]]

    scaler = StandardScaler()
    data = df[[x for x in df.columns if x not in {'timestamp', 'latitude', 'longitude', 'pm25'}]].to_numpy()
    data = scaler.fit_transform(data)
    df[[x for x in df.columns if x not in {'timestamp', 'latitude', 'longitude', 'pm25'}]] = data

    for mnth, split in monthwise_split.items():
        start_time = time.time()

        df_1 = df[df['timestamp'].dt.month == mnth]
        df_1 = df_1.sort_values(by='timestamp')
        df_grouped = df_1.groupby(['latitude', 'longitude'])
        out_file = f'{gnn_data_bihar}/bihar_fw:{FW}_ws:{WS}_{split}_gnn.pkl'

        # gnn_data = {}
        ts = len(df_1['timestamp'].unique())
        num_windows = ts - FW - WS + 2
        gnn_data = [[] for _ in range(num_windows)]
        # print(len(gnn_data))

        for loc, group in df_grouped:

            loc = (loc[0].astype(np.float32), loc[1].astype(np.float32))
            data = group.to_numpy()

            # Since first three columns are timestamp, latitude and longitude respectively
            X, y = data[:, 3:-1], data[:, -1]

            y = np.lib.stride_tricks.sliding_window_view(y, (FW,))
            X = X[:y.shape[0], :]
            X = np.lib.stride_tricks.sliding_window_view(X, (WS, X.shape[1]))
            y = np.lib.stride_tricks.sliding_window_view(y, (WS, y.shape[1]))
            X, y = np.squeeze(X), np.squeeze(y)

            # gnn_data = [[] * X.shape[0]]
            # print(X.shape, y.shape)

            for i, (features, labels) in enumerate(zip(X, y)):
                # print(i, features.shape, labels.shape, len(gnn_data[i]))
                gnn_data[i].append({loc: [features.astype(np.float32), labels.astype(np.float32)]})

        # print(len(gnn_data), len(gnn_data[0]))

        # for data in gnn_data:
        #     print(len(data))

        if Path(out_file).is_file():
            with open(out_file, 'ab') as f:
                pickle.dump(gnn_data, f)
        else:
            with open(out_file, 'wb') as f:
                pickle.dump(gnn_data, f)

        print(f'---------- {mnth} saving complete\t Time: {(time.time()-start_time)/60:.2f} mins ----------')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Define command-line arguments
    parser.add_argument('--fw', help='Forecast Window')
    parser.add_argument('--ws', help='Window Size')

    args = parser.parse_args()

    FW, WS = int(args.fw), int(args.ws)
    monthwise_split = {1: 'test', 5: 'train', 6: 'train', 7: 'train', 8: 'validation', 9: 'train', 10: 'train', 11: 'train', 12: 'train'}

    print(f'---------- NPZ saving start ----------')

    in_file = f'{data_bihar}/bihar_meteo_era5_may_jan_iterative_imputed.pkl'
    # out_file = f'{data_bihar}/bihar_fw:{FW}_ws:{WS}_gnn.npz'

    save_data(in_file, monthwise_split, FW, WS)
    # print(f'---------- NPZ saving complete\t Time: {(time.time()-start_time)/60:.2f} mins ----------')