import numpy as np
from torch.utils import data

class TemporalDataset(data.Dataset):
    def __init__(self, data_fp, forecast_window, start_index, duration):
        self.forecast_window = forecast_window
        self._process_data(data_fp, start_index, duration)

    def _process_location(self, data, index_loc, start_index, duration):

        idx_start, idx_end = start_index+duration-self.forecast_window, start_index+duration

        features = data[idx_start: idx_end, index_loc, :-1]
        pm25_hist = data[start_index:idx_start, index_loc, -1]
        pm25 = data[idx_start:start_index+duration, index_loc, -1]

        return np.float32(features), np.float32(pm25_hist), np.float32(pm25)
        
    def _process_data(self, data_fp, start_index, duration):

        '''
            Shape of the numpy array is [t, l, f]
                t: total number of timestamps
                l: total number of locations
                f: total number of features (including pm25) for each observation
        '''
        data = np.load(data_fp)
        num_locs = data.shape[1]

        self.feature, self.pm25_hist, self.pm25 = [], [], []

        for index_loc in range(num_locs):
            feature, pm25_hist, pm25 = self._process_location(data, index_loc, start_index, duration)
            self.feature.append(feature)
            self.pm25_hist.append(pm25_hist)
            self.pm25.append(pm25)

        self.feature, self.pm25_hist, self.pm25 = np.float32(self.feature), np.float32(self.pm25_hist), np.float32(self.pm25)
    
    def shape(self):
        return self.feature.shape, self.pm25_hist.shape, self.pm25.shape

    def __len__(self):
        assert len(self.pm25_hist) == len(self.feature) == len(self.pm25)
        return len(self.feature)

    def __getitem__(self, idx):
        return self.pm25_hist[idx], self.feature[idx], self.pm25[idx]