import numpy as np
import yaml
import sys
import os
from metpy.units import units
import metpy.calc as mpcalc
import torch
import torch.nn as nn
from models.cells import GRUCell
from torch_geometric.nn import ChebConv, GCNConv, SAGEConv

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_dir)
config_fp = os.path.join(proj_dir, 'config.yaml')

with open(config_fp, 'r') as f:
    config = yaml.safe_load(f)

# ------------- Config parameters start ------------- #
meteo_var = config['meteo_var']
wind_thresh = float(config['threshold']['wind'])
# ------------- Config parameters end   ------------- #


class DGC_GRU(nn.Module):
    def __init__(self, in_dim, hid_dim, city_num, hist_window, forecast_window, batch_size, device, adj_mat, angles):
        super(DGC_GRU, self).__init__()
        self.device = device
        self.adj_mat = torch.Tensor(adj_mat)
        self.angles = torch.Tensor(angles)
        self.hist_window = hist_window
        self.forecast_window = forecast_window
        # self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = 1
        self.gcn_out = 1                            # Should be equal to out_dim
        self.city_num = city_num
        self.batch_size = batch_size

        self.conv = ChebConv(in_dim + self.out_dim, self.gcn_out, K=2)
        self.gru_cell = GRUCell(in_dim + self.out_dim + self.gcn_out, hid_dim)
        self.fc_out = nn.Linear(hid_dim, self.out_dim)

    def generate_edge_indices(self, wind):
        
        m, n = self.adj_mat.size()
        wind_mat = torch.zeros((self.batch_size, m, n)).to(self.device)

        for i in range(m):
            for j in range(n):
                theta_1, theta_2 = self.angles[i, j] - torch.tensor(np.pi/2, dtype=torch.float32), self.angles[i, j]
                wind_mat[:, i, j] = wind[:, i, 1] * torch.cos(theta_1) + wind[:, i, 0] * torch.cos(theta_2)

        wind_mat = torch.where(wind_mat >= wind_thresh, 1, 0)
        adj_mat = self.adj_mat.unsqueeze(0).repeat(self.batch_size, 1, 1)
        
        edges = torch.logical_and(adj_mat, wind_mat)

        edge_indices = []

        for i in range(edges.size(0)):
            r, c = torch.where(edges[i, :, :] == True)
            edges = [(x, y) for x, y in zip(r, c)]

            source_nodes = torch.tensor([edge[0] for edge in edges])
            target_nodes = torch.tensor([edge[1] for edge in edges])
            edge_indices = torch.stack((source_nodes, target_nodes))

        return edge_indices

    def forward(self, feature, pm25_hist):
        '''
            feature shape: (batch_size, forecast_window, number of features)
            pm25_hist shape: (batch_size, T-forecast_window), where T are total number of timestamps in the dataset 
        '''
        self.adj_mat = self.adj_mat.to(self.device)
        feature, pm25_hist = feature.to(self.device), pm25_hist.to(self.device)
        pm25_pred = []

        h0 = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        hn = h0
        xn = pm25_hist[:, -1]

        for i in range(self.forecast_window):
            curr_feature = feature[:, self.hist_window+i]
            x = torch.cat((xn, curr_feature), dim=-1)
            u10 = curr_feature[:, :, meteo_var.index('u10')]
            v10 = curr_feature[:, :, meteo_var.index('v10')]
            wind = torch.stack((u10, v10), axis=-1)

            edge_indices = self.generate_edge_indices(wind).to(self.device)

            # print(curr_feature.size(), u10.size(), v10.size(), wind.size(), self.adj_mat.size(), edge_indices.size())
            x_gcn = x.contiguous()

            x_gcn = x_gcn.view(self.batch_size * self.city_num, -1)
            x_gcn = torch.sigmoid(self.conv(x_gcn, edge_indices))
            x_gcn = x_gcn.view(self.batch_size, self.city_num, -1)

            x = torch.cat((x, x_gcn), dim=-1)
            hn = self.gru_cell(x, hn)

            xn = hn.view(self.batch_size, self.city_num, self.hid_dim)
            xn = self.fc_out(xn)
            pm25_pred.append(xn)

        pm25_pred = torch.stack(pm25_pred, dim=1)
        return pm25_pred