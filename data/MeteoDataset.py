import numpy as np

class MeteoDataset():
    def __init__(self, timestamp, latitude, longitude):
        self.timestamp = timestamp
        self.latitude = latitude
        self.longitude = longitude
        self.rh = None
        self.temp = None
        self.pm25 = None

    def set_features(self, values, option):
        if option == 1:
            self.pm25 = np.array(values, dtype=float)
        elif option == 2:
            self.temp = np.array(values, dtype=float)
        else:
            self.rh = np.array(values, dtype=float)
    
    def __len__(self):
        assert len(self.rh) == len(self.temp) == len(self.pm25) == len(self.timestamp)
        return len(self.timestamp)