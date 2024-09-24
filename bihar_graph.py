import os
import numpy as np
from scipy.spatial import distance
from haversine import haversine, Unit
import torch
from torch_geometric.utils import dense_to_sparse

class Graph():
    def __init__(self, loc_fp, dist_thresh):

        self.locs = self._process_locs(loc_fp)
        self.num_locs = len(self.locs)
        self.dist_thresh = dist_thresh

        self.edge_indices, self.edge_weights, self.edge_attr = self._gen_edge_indices()
        
    def _process_locs(self, loc_fp):
        locs = []

        with open(loc_fp, 'r') as f:
            for line in f:
                data = line.strip().split('|')
                lon, lat = float(data[-2]), float(data[-1])
                locs.append((lon, lat))

        return locs

    def _gen_edge_indices(self):
        dist_mat = distance.cdist(self.locs, self.locs, metric=(lambda u, v : haversine(u, v, unit=Unit.KILOMETERS)))
        angle_mat = distance.cdist(self.locs, self.locs, metric=(lambda u, v : haversine(u, v, unit=Unit.RADIANS)))
        dist_mat = np.where(dist_mat <= self.dist_thresh, dist_mat, 0)
        edge_indices, edge_weights = dense_to_sparse(torch.tensor(dist_mat))
        edge_weights = torch.max(edge_weights) / edge_weights
        edge_weights = edge_weights / torch.max(edge_weights)

        edge_attr = []
        for i in range(edge_indices.size(1)):
            src, dst = edge_indices[0, i], edge_indices[1, i]
            edge_attr.append([dist_mat[src, dst], angle_mat[src, dst]])

        return torch.LongTensor(edge_indices), torch.tensor(edge_weights, dtype=torch.float32),\
            torch.tensor(edge_attr, dtype=torch.float32)