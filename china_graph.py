import os
import sys
import numpy as np
import yaml
from geopy.distance import geodesic
from scipy.spatial import distance
from collections import OrderedDict
# from bresenham import bresenham

proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
config_fp = os.path.join(proj_dir, 'config.yaml')

with open(config_fp, 'r') as f:
    config = yaml.safe_load(f)

# ------------- Config parameters start ------------- #
data_dir = config['dirpath']['data_dir']

loc_fp = data_dir + config['china']['filepath']['locations_fp']
alt_fp = data_dir + config['china']['filepath']['altitude_fp']

DIST_THRESH = config['china']['threshold']['distance']
ALT_THRESH = config['china']['threshold']['altitude']
# ------------- Config parameters end ------------- #

class Graph():
    def __init__(self):
        self.locs = self._process_locs(loc_fp)
        self.num_locs = len(self.locs)
        self.altitude = self._load_altitude()
        self.alt_mat = self._gen_alt_mat()
        self.dist_mat = self._gen_dist_mat()

    def _load_altitude(self):
        assert os.path.isfile(alt_fp)
        altitude = np.load(alt_fp)
        return altitude
    
    def _lonlat2xy(self, lon, lat, is_aliti):
        if is_aliti:
            lon_l = 100.0
            lon_r = 128.0
            lat_u = 48.0
            lat_d = 16.0
            res = 0.05
        else:
            lon_l = 103.0
            lon_r = 122.0
            lat_u = 42.0
            lat_d = 28.0
            res = 0.125
        x = np.int64(np.round((lon - lon_l - res / 2) / res))
        y = np.int64(np.round((lat_u + res / 2 - lat) / res))
        return x, y
        
    def _process_locs(self, loc_fp):
        locs = []

        with open(loc_fp, 'r') as f:
            
            for line in f:
                data = line.strip().split('|')
                lon, lat = float(data[-2]), float(data[-1])
                x, y = self._lonlat2xy(lon, lat, True)
                locs.append([x, y])

        return locs
    
    def _gen_alt_mat(self):
        pass

    def _gen_dist_mat(self):
        dist_mat = distance.cdist(self.locs, self.locs, 'euclidean')
        # dist_mat = np.where(dist_mat <= DIST_THRESH, True, False)
        return dist_mat

    
if __name__ == '__main__':
    g = Graph()
    
    # print(g.num_locs)
    # print(g.altitude.shape)

    # print(g.altitude)
    # print(g.dist_mat)

    dist_mat = g.dist_mat.flatten()

    # Define the bins
    bins = np.arange(0, 500, 50)

    # Digitize the matrix elements into bins
    bin_indices = np.digitize(dist_mat, bins, right=False)

    # Count the frequency of each bin
    bin_counts = np.bincount(bin_indices, minlength=len(bins)+1)

    # Frequency of elements in each range
    # bin_counts[1:] skips the count for the out-of-range elements (if any)
    for i in range(1, len(bins)):
        print(f"Frequency of elements in range {bins[i-1]}-{bins[i]}: {bin_counts[i]}")