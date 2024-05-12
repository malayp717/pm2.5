import numpy as np
import torch
import torch.nn as nn
from models.cells import GRUCell
from torch_geometric.nn import ChebConv, GCNConv, SAGEConv, GATConv

class Seq2Seq_GC_GRU(nn.Module):
    def __init__(self, in_dim, hid_dim, city_num, hist_window, forecast_window, batch_size, device, adj_mat):
        super(Seq2Seq_GC_GRU, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.city_num = city_num
        self.edge_indices = self._process_adj_mat(adj_mat)
        self.hist_window = hist_window
        self.forecast_window = forecast_window

        self.hid_dim = hid_dim
        self.out_dim = 1
        self.gcn_out = 1                            # Should be equal to out_dim

        self.gru_cell_hist = GRUCell(2, self.hid_dim)
        self.fc_hist = nn.Linear(self.hid_dim, self.out_dim)
        self.conv = ChebConv(in_dim+self.out_dim, self.gcn_out, K=2)
        self.gru_cell = GRUCell(in_dim + self.out_dim + self.gcn_out, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

    def _process_adj_mat(self, adj_mat):

        r, c = np.where(adj_mat == True)
        edges = [(x, y) for x, y in zip(r, c)]

        source_nodes = torch.tensor(np.array([edge[0] for edge in edges]))
        target_nodes = torch.tensor(np.array([edge[1] for edge in edges]))

        edge_indices = torch.LongTensor(torch.stack((source_nodes, target_nodes)))

        edge_indices = edge_indices.view(2, 1, -1).repeat(1, self.batch_size, 1) + torch.arange(self.batch_size).view(1, -1, 1) * self.city_num
        edge_indices = edge_indices.view(2, -1)

        return edge_indices

    def forward(self, feature, pm25_hist):
        '''
            feature shape: (batch_size, forecast_window, number of features)
            pm25_hist shape: (batch_size, T-forecast_window), where T are total number of timestamps in the dataset 
        '''
        self.edge_indices = self.edge_indices.to(self.device)

        feature, pm25_hist = feature.to(self.device), pm25_hist.to(self.device)
        pm25_pred = []

        '''
            PM2.5 history embedding implementation
        '''
        h0 = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        hn = h0
        xn = torch.zeros(self.batch_size, self.city_num, self.out_dim).to(self.device)

        for i in range(self.hist_window):
            x = torch.cat((xn, pm25_hist[:, i]), dim=-1)
            x = x.contiguous()

            hn = self.gru_cell_hist(x, hn)
            xn = hn.view(self.batch_size, self.city_num, self.hid_dim)
            xn = self.fc_hist(xn)

        '''
            Current Forecast Window implementation
        '''
        hn = hn.view(self.batch_size * self.city_num, self.hid_dim)

        for i in range(self.forecast_window):
            x = torch.cat((xn, feature[:, self.hist_window+i]), dim=-1)
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