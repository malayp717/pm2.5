import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import torch
import torch_geometric
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch_geometric.data import Data
from geopy.distance import geodesic
from metpy.units import units
import metpy.calc as calc
from constants import *
from utils import *

class Node():
    def __init__(self, features, label, loc):
        self.loc = loc
        self.features = features
        self.label = label

class Edge():
    def __init__(self, source, target):
        self.source = source
        self.target = target

class GraphDataset():
    def __init__(self):
        pass

def distance_matrix(locs):

    dist = distance.pdist(locs, lambda u, v: geodesic(u, v).kilometers)
    num_nodes = len(locs)

    # num_dist = (num_nodes * (num_nodes-1))//2
    # print(num_nodes, num_dist, len(dist))
    
    dist_u, dist_l = np.zeros((num_nodes, num_nodes)), np.zeros((num_nodes, num_nodes))

    mask = np.triu_indices(num_nodes, k=1)

    dist_u[mask] = dist
    dist_l = dist_u.T

    return dist_u + dist_l 


if __name__ == '__main__':
    
    train_locs = load_locs_as_tuples(f'{data_bihar}/train_locations.txt')
    val_locs = load_locs_as_tuples(f'{data_bihar}/val_locations.txt')
    test_locs = load_locs_as_tuples(f'{data_bihar}/test_locations.txt')

    data_file = f'{data_bihar}/bihar_512_sensor_era5_rnn.pkl'
    df = pd.read_pickle(data_file)
    locs, nodes, edges = [], [], []

    scaler = StandardScaler()
    data = df[[x for x in df.columns if x not in {'timestamp', 'latitude', 'longitude', 'pm25'}]].to_numpy()
    data = scaler.fit_transform(data)
    df[[x for x in df.columns if x not in {'timestamp', 'latitude', 'longitude', 'pm25'}]] = data

    df_grouped = df.groupby(['latitude', 'longitude'])
    for loc, group in df_grouped:

        loc = (loc[0].astype(np.float32), loc[1].astype(np.float32))
        locs.append(loc)
        data = group.to_numpy()
        # Since first three columns are timestamp, latitude and longitude respectively
        X, y = data[:, 3:-1], data[:, -1]

        nodes.append(Node(X.astype(np.float32), y.astype(np.float32), loc))

    dist_mat = distance_matrix(locs)

    cond = np.logical_and(np.where(dist_mat <= 100, True, False), np.where(dist_mat > 0, True, False)) 
    dist_mat = np.logical_and(cond, dist_mat)

    r, c = np.where(dist_mat == True)

    edges = [Edge(x, y) for x, y in zip(r, c)]

    node_features_tensor = torch.tensor(np.array([node.features for node in nodes]))
    node_labels_tensor = torch.tensor(np.array([node.label for node in nodes]))
    
    source_nodes_tensor = torch.tensor(np.array([edge.source for edge in edges]))
    target_nodes_tensor = torch.tensor(np.array([edge.target for edge in edges]))

    graph = Data(x=node_features_tensor, y=node_labels_tensor, edge_index=[source_nodes_tensor, target_nodes_tensor])

    print(graph)

    print(node_features_tensor.shape, node_labels_tensor.shape, source_nodes_tensor.shape, target_nodes_tensor.shape)