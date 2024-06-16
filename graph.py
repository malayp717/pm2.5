import os
import numpy as np
from scipy.spatial import distance
from bresenham import bresenham
from utils import haversine_dist
import torch
from torch_geometric.utils import dense_to_sparse, to_dense_adj

class Graph():
    def __init__(self, location, loc_fp, dist_thresh, alt_fp=None, alt_thresh=None):

        if location == 'china':
            assert alt_fp is not None
            assert alt_thresh is not None

        self.locs = self._process_locs(loc_fp)
        self.num_locs = len(self.locs)
        self.dist_thresh = dist_thresh
        self.alt_thresh = alt_thresh

        self.edge_indices = self._gen_edge_indices()

        if location == 'china':
            self.altitude = np.load(alt_fp)
            self.edge_indices = self._update_edge_indices()
        
        self.adj_mat = to_dense_adj(torch.tensor(self.edge_indices)).squeeze()
    
    def _lonlat2xy(self, lon, lat, is_alti):
        if is_alti:
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
                locs.append((lon, lat))

        return locs

    def _gen_edge_indices(self):
        dist_mat = distance.cdist(self.locs, self.locs, metric=(lambda u, v : haversine_dist(u, v)))
        dist_mat = np.where(dist_mat <= self.dist_thresh, True, False)
        edge_indices, _ = dense_to_sparse(torch.tensor(dist_mat))
        return edge_indices
    
    def _update_edge_indices(self):
        edge_indices = []

        for i in range(self.edge_indices.shape[1]):
            src, dst = self.edge_indices[0, i], self.edge_indices[1, i]
            src_lon, src_lat = self.locs[src][0], self.locs[src][1]
            dst_lon, dst_lat = self.locs[dst][0], self.locs[dst][1]

            src_x, src_y = self._lonlat2xy(src_lon, src_lat, True)
            dst_x, dst_y = self._lonlat2xy(dst_lon, dst_lat, True)
            
            points = np.asarray(list(bresenham(src_y, src_x, dst_y, dst_x))).transpose((1,0))
            altitude_points = self.altitude[points[0], points[1]]
            altitude_src = self.altitude[src_y, src_x]
            altitude_dest = self.altitude[dst_y, dst_x]
            if np.sum(altitude_points - altitude_src > self.alt_thresh) < 3 and \
               np.sum(altitude_points - altitude_dest > self.alt_thresh) < 3:
                edge_indices.append(self.edge_indices[:,i])
            
        return np.stack(edge_indices, axis=1)