import numpy as np
# import pandas as pd
from scipy.spatial import distance
# from sklearn.preprocessing import StandardScaler
import torch
from geopy.distance import geodesic
# from metpy.units import units
# import h5py
# from pathlib import Path
from torch_geometric.data import Data, Dataset
import pickle
from constants import *

class Node():
    def __init__(self, features, label, loc):
        self.loc = loc
        self.features = features
        self.label = label

class Edge():
    def __init__(self, source, target):
        self.source = source
        self.target = target 

class SpatioTemporalDataset(Dataset):

    def distance_matrix(self, locs):
        dist = distance.pdist(locs, lambda u, v: geodesic(u, v).kilometers)
        num_nodes = len(locs)
        
        dist_u, dist_l = np.zeros((num_nodes, num_nodes)), np.zeros((num_nodes, num_nodes))
        mask = np.triu_indices(num_nodes, k=1)

        dist_u[mask] = dist
        dist_l = dist_u.T

        return dist_u + dist_l
    
    def generate_edges(self, locs):
        dist_mat = self.distance_matrix(locs)
        cond = np.logical_and(np.where(dist_mat <= DIST_THRESH, True, False), np.where(dist_mat > 0, True, False)) 
        dist_mat = np.logical_and(cond, dist_mat)

        r, c = np.where(dist_mat == True)
        edges = [Edge(x, y) for x, y in zip(r, c)]

        source_nodes_tensor = torch.tensor(np.array([edge.source for edge in edges]))
        target_nodes_tensor = torch.tensor(np.array([edge.target for edge in edges]))

        return source_nodes_tensor, target_nodes_tensor
    
    def get_input_dimensions(self):
        if len(self.data) == 0:
            return 0, 0, 0
        
        row = self.data[0]
        X, y = None, None

        for ele in row:
            for _, item in ele.items():
                X, y = item[0], item[1]
                break
            break

        return X.shape[-1], self.hidden_dim, y.shape[-1]

    def __init__(self, file, locs, hidden_dim):
        with open(file, 'rb') as f:
            self.data = pickle.load(f)

        self.locs = locs
        self.hidden_dim = hidden_dim
        self.source_nodes, self.target_nodes = self.generate_edges(self.locs)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        nodes = []

        for ele in row:
            for loc, item in ele.items():
                X, y = item[0], item[1]
                nodes.append(Node(X.astype(np.float32), y.astype(np.float32), loc))
                break

        node_features = torch.tensor(np.array([node.features for node in nodes]))
        node_labels = torch.tensor(np.array([node.label for node in nodes]))

        return node_features, torch.stack((self.source_nodes, self.target_nodes)), node_labels
        # return Data(x=node_features, edge_index=torch.stack((self.source_nodes, self.target_nodes)), y=node_labels)