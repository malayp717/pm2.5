import numpy as np
import math
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
        self._generate_edges()
        self.angles = self._generate_angles()
        self.angles = np.float32(self.angles)

    def _process_locs(self, loc_fp):
        locs = []

        with open(loc_fp, 'r') as f:
            
            for line in f:
                data = line.strip().split('|')
                locs.append([float(data[-2]), float(data[-1])])

        return locs
    
    def angle_from_coord(self, loc_1, loc_2):
        # loc: [longitude, latitude]
        dLon = (loc_2[0] - loc_1[0])

        y = math.sin(dLon) * math.cos(loc_2[1])
        x = math.cos(loc_1[1]) * math.sin(loc_2[1]) - math.sin(loc_1[1]) * math.cos(loc_2[1]) * math.cos(dLon)

        brng = math.atan2(y, x)

        '''
            uncomment these lines for angle in degrees
        '''
        # brng = math.degrees(brng)
        # brng = (brng + 360) % 360
        # brng = 360 - brng                       # count degrees clockwise - remove to make counter-clockwise

        return brng
    
    def _generate_angles(self):
        angles = np.zeros((len(self.locs), len(self.locs)))

        for i, loc_x in enumerate(self.locs):
            for j, loc_y in enumerate(self.locs):
                if i == j: continue

                angles[i, j] = self.angle_from_coord(loc_x, loc_y)

        return angles

    def _distance_matrix(self, locs):
        dist = distance.pdist(locs, lambda u, v: geodesic(u, v).kilometers)
        num_nodes = len(locs)
        
        dist_u, dist_l = np.zeros((num_nodes, num_nodes)), np.zeros((num_nodes, num_nodes))
        mask = np.triu_indices(num_nodes, k=1)

        dist_u[mask] = dist
        dist_l = dist_u.T

        return dist_u + dist_l
    
    def _generate_edges(self):
        self.adj_mat = self._distance_matrix(self.locs)
        cond = np.logical_and(np.where(self.adj_mat <= DIST_THRESH, True, False), np.where(self.adj_mat > 0, True, False)) 
        self.adj_mat = np.logical_and(cond, self.adj_mat)

if __name__ == '__main__':
    graph = Graph(loc_fp)
    print(graph.num_locs)
    print(graph.edge_indices)