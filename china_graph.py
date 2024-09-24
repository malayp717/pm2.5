import os
import sys
import yaml
import numpy as np
from scipy.spatial import distance
from haversine import haversine, Unit
import torch
from torch_geometric.utils import dense_to_sparse
from bresenham import bresenham

proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
config_fp = os.path.join(proj_dir, 'china_config.yaml')

with open(config_fp, 'r') as f:
    config = yaml.safe_load(f)

# ------------- Config parameters start ------------- #
data_dir = config['dirpath']['data_dir']
model_dir = config['dirpath']['model_dir']

npy_fp = data_dir + config['filepath']['npy_fp']
loc_fp = data_dir + config['filepath']['locations_fp']
alt_fp = data_dir + config['filepath']['altitude_fp']

update = int(config['dataset']['update'])
data_start = config['dataset']['data_start']
data_end = config['dataset']['data_end']

dist_thresh = float(config['threshold']['distance'])
alt_thresh = float(config['threshold']['altitude'])
haze_thresh = float(config['threshold']['haze'])
# ------------- Config parameters end   ------------- #

class Graph:
    def __init__(self, loc_fp, alt_fp, dist_thresh, alt_thresh):
        self.locs = self._process_locs(loc_fp)
        self.num_locs = len(self.locs)
        self.dist_thresh = dist_thresh
        self.alt_thresh = alt_thresh
        self.edge_indices, self.edge_weights, self.edge_attr = self._gen_edge_indices()

        self.altitude = self._load_altitude(alt_fp)
        self.edge_indices, self.edge_weights, self.edge_attr = self._update_edges()
        self.edge_indices, self.edge_weights, self.edge_attr = torch.tensor(self.edge_indices, dtype=torch.long),\
            torch.tensor(self.edge_weights, dtype=torch.float32), torch.tensor(self.edge_attr, dtype=torch.float32)


    def _load_altitude(self, alt_fp):
        altitude = np.load(alt_fp)
        return altitude
        
    def _process_locs(self, loc_fp):
        locs = []
        with open(loc_fp, 'r') as f:
            for line in f:
                data = line.strip().split('|')
                lon, lat = float(data[2]), float(data[3])
                locs.append((lat, lon))  # Note: haversine expects (lat, lon) order
        return locs

    def _gen_edge_indices(self):
        dist_mat = distance.cdist(self.locs, self.locs, metric=lambda u, v: haversine(u, v, unit=Unit.KILOMETERS))
        angle_mat = distance.cdist(self.locs, self.locs, metric=lambda u, v: haversine(u, v, unit=Unit.RADIANS))
        dist_mat = np.where(dist_mat <= self.dist_thresh, dist_mat, 0)
        edge_indices, edge_weights = dense_to_sparse(torch.tensor(dist_mat))
        edge_indices, edge_weights = edge_indices.numpy(), edge_weights.numpy()
        edge_weights = np.min(edge_weights) / edge_weights

        edge_attr = []
        for i in range(edge_indices.shape[1]):
            src, dst = edge_indices[0, i], edge_indices[1, i]
            edge_attr.append([dist_mat[src, dst], angle_mat[src, dst]])

        return np.array(edge_indices, dtype=np.int64), np.array(edge_weights, dtype=np.float32),\
            np.array(edge_attr, dtype=np.float32)
    
    def _lonlat2xy(self, lon, lat):
        lon_l, lon_r = 100.0, 128.0
        lat_u, lat_d = 48.0, 16.0
        res = 0.05
        x = np.int64(np.round((lon - lon_l - res / 2) / res))
        y = np.int64(np.round((lat_u + res / 2 - lat) / res))
        return x, y
    
    def _update_edges(self):
        edge_indices, edge_attr, edge_weights = [], [], []

        for i in range(self.edge_indices.shape[1]):
            src, dest = self.edge_indices[0, i], self.edge_indices[1, i]
            src_lat, src_lon = self.locs[src][0], self.locs[src][1]
            dest_lat, dest_lon = self.locs[dest][0], self.locs[dest][1]

            src_x, src_y = self._lonlat2xy(src_lon, src_lat)
            dest_x, dest_y = self._lonlat2xy(dest_lon, dest_lat)

            points = np.asarray(list(bresenham(src_y, src_x, dest_y, dest_x))).transpose((1,0))
            altitude_points = self.altitude[points[0], points[1]]
            altitude_src = self.altitude[src_y, src_x]
            altitude_dest = self.altitude[dest_y, dest_x]

            if np.sum(altitude_points - altitude_src > self.alt_thresh) < 3 and \
               np.sum(altitude_points - altitude_dest > self.alt_thresh) < 3:
                edge_indices.append(self.edge_indices[:,i])
                edge_attr.append(self.edge_attr[i])
                edge_weights.append(self.edge_weights[i])

        edge_indices = np.stack(edge_indices, axis=1)
        edge_attr = np.stack(edge_attr, axis=0)
        edge_weights = np.stack(edge_weights, axis=0)

        return edge_indices, edge_weights, edge_attr

if __name__ == '__main__':
    graph = Graph(loc_fp, alt_fp, dist_thresh, alt_thresh)
    print(graph.edge_indices.max())