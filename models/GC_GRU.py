import numpy as np
import torch
import torch.nn as nn
from models.cells import GRUCell
from torch_geometric.nn import ChebConv, GCNConv, SAGEConv

class GC_GRU(nn.Module):
    def __init__(self, in_dim, hid_dim, city_num, hist_window, forecast_window, batch_size, device, edge_indices):
        super(GC_GRU, self).__init__()
        self.device = device
        self.edge_indices = torch.LongTensor(edge_indices)
        self.edge_indices = self.edge_indices.view(2, 1, -1).repeat(1, batch_size, 1) + torch.arange(batch_size).view(1, -1, 1) * city_num
        self.edge_indices = self.edge_indices.view(2, -1)
        self.hist_window = hist_window
        self.forecast_window = forecast_window
        # self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = 1
        self.gcn_out = 1                            # Should be equal to out_dim
        self.city_num = city_num
        self.batch_size = batch_size

        # self.conv = ChebConv(in_dim+self.out_dim, self.gcn_out, K=2)
        # self.conv = GCNConv(in_dim+self.out_dim, self.gcn_out, add_self_loops=True)
        self.conv = SAGEConv(in_dim + self.out_dim, self.gcn_out)
        self.gru_cell = GRUCell(in_dim + self.out_dim + self.gcn_out, hid_dim)
        self.fc_out = nn.Linear(hid_dim, self.out_dim)

    def forward(self, feature, pm25_hist):
        '''
            feature shape: (batch_size, forecast_window, number of features)
            pm25_hist shape: (batch_size, T-forecast_window), where T are total number of timestamps in the dataset 
        '''
        self.edge_indices = self.edge_indices.to(self.device)

        feature, pm25_hist = feature.to(self.device), pm25_hist.to(self.device)
        pm25_pred = []

        h0 = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        hn = h0
        xn = pm25_hist[:, -1]

        for i in range(self.forecast_window):
            x = torch.cat((xn, feature[:, self.hist_window+i]), dim=-1)
            # x = self.fc_in(x)
            x_gcn = x.contiguous()

            x_gcn = x_gcn.view(self.batch_size * self.city_num, -1)
            x_gcn = torch.sigmoid(self.conv(x_gcn, self.edge_indices))
            x_gcn = x_gcn.view(self.batch_size, self.city_num, -1)

            x = torch.cat((x, x_gcn), dim=-1)
            hn = self.gru_cell(x, hn)

            xn = hn.view(self.batch_size, self.city_num, self.hid_dim)
            xn = self.fc_out(xn)
            pm25_pred.append(xn)

        pm25_pred = torch.stack(pm25_pred, dim=1)
        return pm25_pred