import numpy as np
from datetime import datetime
from torch.utils import data

class SpatioTemporalDataset(data.Dataset):
    def __init__(self, data_fp, forecast_window, hist_window, start_date, end_date, data_start, update, edge_indices):
        '''
            Shape of the numpy array is [t, l, f]
                t: total number of timestamps
                l: total number of locations
                f: total number of features (including pm25) for each observation
        '''
        self.npy_data = np.load(data_fp)
        self.edge_indices = edge_indices
        self.forecast_window = forecast_window
        self.hist_window = hist_window
        self.start_idx, self.end_idx = self._get_indices(start_date, end_date, data_start, update)
        self._norm()
        self.pm25 = np.expand_dims(self.pm25, axis=-1)
        self._process_data()
        self.feature = np.float32(self.feature)
        self.pm25 = np.float32(self.pm25)

    def _get_indices(self, start_date, end_date, data_start, update):
        start_idx = (datetime(*start_date) - datetime(*data_start)).days * (24//update)
        end_idx = (datetime(*end_date) - datetime(*data_start)).days * (24//update)
        return start_idx, end_idx
        
    def _norm(self):
        overall_mean, overall_std = np.mean(self.npy_data, axis=(0,1)), np.std(self.npy_data, axis=(0,1))

        self.feature_mean, self.feature_std = overall_mean[:-1], overall_std[:-1]
        self.pm25_mean, self.pm25_std = overall_mean[-1], overall_std[-1]

        data = (self.npy_data[self.start_idx:self.end_idx+1]-overall_mean) / overall_std
        self.feature, self.pm25 = data[:, :, :-1], data[:, :, -1]

    def _add_t(self, arr):
        # Total Time steps >= forecast_window + hist_window
        assert arr.shape[0] > self.forecast_window + self.hist_window

        seq_len = self.forecast_window+self.hist_window
        arr_ts = []

        for i in range(seq_len, arr.shape[0]):
            arr_ts.append(arr[i-seq_len:i])

        arr_ts = np.stack(arr_ts, axis=0)
        return arr_ts

    def _process_data(self):
        self.feature = self._add_t(self.feature)
        self.pm25 = self._add_t(self.pm25)

    def __len__(self):
        assert len(self.feature) == len(self.pm25)
        return len(self.feature)

    def __getitem__(self, idx):
        return self.feature[idx], self.pm25[idx]