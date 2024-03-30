import numpy as np
# from scipy.spatial import distance
import time
# import matplotlib.pyplot as plt
# import networkx as nx
import torch
import torch.nn as nn
# import torch_geometric
# from sklearn.preprocessing import StandardScaler
# from torch_geometric.utils import dense_to_sparse, to_dense_adj, degree
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
# from torchinfo import summary
# from geopy.distance import geodesic
# from metpy.units import units
# import metpy.calc as calc
from constants import *
from utils import *
from dataset.SpatioTemporalDataset import loadSpatioTemporalData
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device):
        super(GNNLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device

        self.embedding = nn.Linear(self.input_dim, self.hidden_dim)
        self.lstm_cell = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.conv = GCNConv(self.hidden_dim, self.hidden_dim, add_self_loops=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x, edge_index):
        h, c = torch.zeros(x.size(0), self.hidden_dim), torch.zeros(x.size(0), self.hidden_dim)
        h, c = h.to(self.device), c.to(self.device)

        for t in range(x.size(1)):
            input_features = self.embedding(x[:, t, :])
            h, c = self.lstm_cell(input_features, (h, c))
            h = self.conv(h, edge_index)

        out = self.fc(h)
        return out

if __name__ == '__main__':
    
    # train_locs = load_locs_as_tuples(f'{data_bihar}/train_locations.txt')
    # val_locs = load_locs_as_tuples(f'{data_bihar}/val_locations.txt')
    # test_locs = load_locs_as_tuples(f'{data_bihar}/test_locations.txt')

    DIST_THRESH, FW, BATCH_SIZE, LR, NUM_EPOCHS = 100, 12, 512, 5e-2, 10

    data_file = f'{data_bihar}/bihar_meteo_era5_may_jan_knn_imputed.pkl'

    locs, node_features, node_labels, source_nodes, target_nodes = loadSpatioTemporalData(data_file, FW, DIST_THRESH)
    dataset = [Data(x=node_features, edge_index=torch.stack((source_nodes, target_nodes)), y=node_labels)]

    input_dim, hidden_dim, output_dim = node_features.size(-1), 64, node_labels.size(-1)
    print(input_dim, hidden_dim, output_dim)

    model = GNNLSTM(input_dim, hidden_dim, output_dim, device)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    loader = DataLoader(dataset, batch_size=1)
    # losses = []

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        for data in loader:
            x, y, edge_index = data.x.to(device), data.y[:, -1, :].to(device), data.edge_index.to(device)

            preds = model(x, edge_index)
            
            loss = torch.sqrt(criterion(y, preds))
            loss.backward()
            optimizer.step()

            # losses.append(loss.item())

        print(f'Epoch: {epoch+1}/{NUM_EPOCHS}, train_loss: {loss.item():.4f},\
            time_taken: {(time.time()-start_time)/60:.2f} mins')