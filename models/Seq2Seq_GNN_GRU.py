import numpy as np
import torch
import torch.nn as nn
from models.cells import GRUCell
from torch_geometric.nn import ChebConv, GraphConv
from torch_geometric.utils import dense_to_sparse

'''
    Encoder part for the PM2.5 History implementation
'''
class Encoder(nn.Module):
    def __init__(self, in_dim, hid_dim, city_num, hist_len, batch_size, device, adj_mat):
        super(Encoder, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = 1

        self.city_num = city_num
        self.hist_len = hist_len
        self.batch_size = batch_size

        self.device = device
        self.edge_indices, self.edge_weights = self._process_adj_mat(adj_mat)

        # self.conv = ChebConv(self.in_dim + 2*self.out_dim, self.hid_dim, K=2)
        self.conv = GraphConv(self.in_dim + 2*self.out_dim, self.hid_dim)
        self.gru_cell = GRUCell(self.in_dim + 2*self.out_dim + self.hid_dim, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

    def _process_adj_mat(self, adj_mat):
        # edge_indices, _ = dense_to_sparse(adj_mat.clone().detach())
        # edge_indices = edge_indices.view(2, 1, -1).repeat(1, self.batch_size, 1) + torch.arange(self.batch_size).view(1, -1, 1) * self.city_num
        # edge_indices = edge_indices.view(2, -1)

        adj_mat = adj_mat.repeat(self.batch_size, 1, 1)
        edge_indices, edge_weights = dense_to_sparse(adj_mat)

        return edge_indices, edge_weights
    
    def forward(self, X, y):

        '''
            X shape: [batch_size, hist_len+forecast_len, city_num, num_features]
            y shape: [batch_size, hist_len+forecast_len, city_num, 1]
        '''

        self.edge_indices, self.edge_weights = self.edge_indices.to(self.device), self.edge_weights.to(self.device)
        X, y = X.to(self.device), y.to(self.device)

        h0 = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        hn = h0
        xn = torch.zeros(self.batch_size, self.city_num, self.out_dim).to(self.device)
        H, preds = [], []

        for i in range(self.hist_len):
            x = torch.cat((xn, y[:, i], X[:, i]), dim=-1)
            x_gcn = x.contiguous()

            x_gcn = x_gcn.view(self.batch_size * self.city_num, -1)
            x_gcn = torch.sigmoid(self.conv(x=x_gcn, edge_index=self.edge_indices, edge_weight=self.edge_weights))
            x_gcn = x_gcn.view(self.batch_size, self.city_num, -1)

            x = torch.cat((x, x_gcn), dim=-1)
            hn = self.gru_cell(x, hn)

            xn = hn.view(self.batch_size, self.city_num, self.hid_dim)
            xn = self.fc_out(xn)

            H.append(hn.view(self.batch_size, self.city_num, self.hid_dim))
            preds.append(xn)

        preds = torch.stack(preds, dim=1)
        H = torch.stack(H, dim=1)
        return H, preds

'''
    Decoder part for the PM2.5 Forecasting implementation
'''
class Decoder(nn.Module):
    def __init__(self, in_dim, hid_dim, city_num, hist_len, forecast_len, batch_size, device):
        super(Decoder, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = 1

        self.city_num = city_num
        self.hist_len = hist_len
        self.forecast_len = forecast_len
        self.batch_size = batch_size

        self.device = device

        self.fc_in = nn.Linear(self.in_dim + self.out_dim, self.hid_dim)
        # self.gru_cell = GRUCell(self.in_dim + self.out_dim, self.hid_dim)
        # self.gru_cell = GRUCell(self.in_dim + self.hid_dim + self.out_dim, self.hid_dim)
        self.gru_cell = GRUCell(self.hid_dim, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

    def forward(self, X, H, xn):

        ''' 
            X shape: [batch_size, hist_len+forecast_len, city_num, num_features]
            H shape: [batch_size, hist_len, city_num, hid_dim]
            xn shape: [batch_size, city_num, out_dim]
        '''
        X, H, xn = X.to(self.device), H.to(self.device), xn.to(self.device)
        
        h0 = H.view(self.batch_size * self.city_num, self.hid_dim)
        hn = h0
        preds = []

        for i in range(self.forecast_len):
            x = torch.cat((xn, X[:, self.hist_len+i, :, -3:]), dim=-1)
            x = x.contiguous()

            x = self.fc_in(x)
            hn = self.gru_cell(x, hn)

            xn = hn.view(self.batch_size, self.city_num, self.hid_dim)
            xn = self.fc_out(xn)

            preds.append(xn)
        
        preds = torch.stack(preds, dim=1)
        return preds

class Seq2Seq_GNN_GRU(nn.Module):
    def __init__(self, in_dim_enc, in_dim_dec, hid_dim, city_num, hist_len, forecast_len, batch_size, device, adj_mat):
        super(Seq2Seq_GNN_GRU, self).__init__()

        self.batch_size = batch_size
        self.city_num = city_num
        self.hid_dim = hid_dim
        self.out_dim = 1
        self.hist_len = hist_len

        self.Encoder = Encoder(in_dim_enc, hid_dim, city_num, hist_len, batch_size, device, adj_mat)
        self.Decoder = Decoder(in_dim_dec, hid_dim, city_num, hist_len, forecast_len, batch_size, device)

    def forward(self, X, y):
        
        H, xn = self.Encoder(X, y)
        print(H.size(), xn.size())
        H, xn = H[:, -1], xn[:, -1]
        H, xn = H.contiguous(), xn.contiguous()

        preds = self.Decoder(X, H, xn)

        return preds