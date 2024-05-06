import numpy as np
import yaml
import os
import sys
from geopy.distance import geodesic
from scipy.spatial import distance
import torch


proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
config_fp = os.path.join(proj_dir, 'config.yaml')

with open(config_fp, 'r') as f:
    config = yaml.safe_load(f)

# ------------- Config parameters start ------------- #
data_dir = config['filepath']['data_dir']
loc_fp = data_dir + config['filepath']['bihar_locations_fp']

DIST_THRESH = int(config['threshold']['distance'])
# ------------- Config parameters end   ------------- #

class Graph():
    def __init__(self, loc_fp):
        self.locs = self._process_locs(loc_fp)
        self.num_locs = len(self.locs)
        self.edge_indices = self._generate_edges(self.locs)

    def _process_locs(self, loc_fp):
        locs = []

        with open(loc_fp, 'r') as f:
            
            for line in f:
                data = line.strip().split('|')
                locs.append([data[-2], data[-1]])

        return locs

    def _distance_matrix(self, locs):
        dist = distance.pdist(locs, lambda u, v: geodesic(u, v).kilometers)
        num_nodes = len(locs)
        
        dist_u, dist_l = np.zeros((num_nodes, num_nodes)), np.zeros((num_nodes, num_nodes))
        mask = np.triu_indices(num_nodes, k=1)

        dist_u[mask] = dist
        dist_l = dist_u.T

        return dist_u + dist_l
    
    def _generate_edges(self, locs):
        dist_mat = self._distance_matrix(locs)
        cond = np.logical_and(np.where(dist_mat <= DIST_THRESH, True, False), np.where(dist_mat > 0, True, False)) 
        dist_mat = np.logical_and(cond, dist_mat)

        r, c = np.where(dist_mat == True)
        edges = [(x, y) for x, y in zip(r, c)]

        source_nodes = torch.tensor(np.array([edge[0] for edge in edges]))
        target_nodes = torch.tensor(np.array([edge[1] for edge in edges]))

        return torch.stack((source_nodes, target_nodes))

if __name__ == '__main__':
    graph = Graph(loc_fp)
    print(graph.num_locs)
    print(graph.edge_indices)