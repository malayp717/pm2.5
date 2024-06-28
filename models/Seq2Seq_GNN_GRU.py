import numpy as np
import torch
import torch.nn as nn
from models.cells import GRUCell
from models.Attention import Attention
from torch_geometric.nn import ChebConv, GraphConv
from torch_geometric.utils import dense_to_sparse

'''
    Encoder part for the PM2.5 History implementation
'''
class Encoder(nn.Module):
    def __init__(self, in_dim, hid_dim, city_num, num_embeddings, hist_len, batch_size, device, adj_mat):
        super(Encoder, self).__init__()
        self.emb_dim = 16
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = 1

        self.city_num = city_num
        self.hist_len = hist_len
        self.batch_size = batch_size

        self.device = device
        self.edge_indices, self.edge_weights = self._process_adj_mat(adj_mat)

        self.sptemp_emb = nn.Embedding(num_embeddings, embedding_dim=self.emb_dim)
        self.conv = GraphConv(self.in_dim - 1 + self.emb_dim + 2*self.out_dim, self.hid_dim)
        self.gru_cell = GRUCell(self.in_dim -1 + self.emb_dim + 2*self.out_dim + self.hid_dim, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

    def _process_adj_mat(self, adj_mat):

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
            emb = self.sptemp_emb(X[:, i, :, -1].long())
            x = torch.cat((xn, y[:, i], X[:, i, :, :-1], emb), dim=-1)
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
    def __init__(self, in_dim, hid_dim, city_num, num_embeddings, hist_len, forecast_len, batch_size, device):
        super(Decoder, self).__init__()
        self.emb_dim = 16
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = 1

        self.city_num = city_num
        self.hist_len = hist_len
        self.forecast_len = forecast_len
        self.batch_size = batch_size

        self.device = device

        self.sptemp_emb = nn.Embedding(num_embeddings, embedding_dim=self.emb_dim)
        # self.fc_in = nn.Linear(self.emb_dim + self.out_dim, self.hid_dim)
        self.gru_cell = GRUCell(self.emb_dim + self.out_dim, self.hid_dim)
        # self.gru_cell = GRUCell(self.emb_dim + self.out_dim + self.hid_dim, self.hid_dim)
        self.attn = Attention(self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

    def forward(self, X, H, xn):

        ''' 
            X shape: [batch_size, hist_len+forecast_len, city_num, num_features]
            H shape: [batch_size, hist_len, city_num, hid_dim]
            xn shape: [batch_size, city_num, out_dim]
        '''
        X, H, xn = X.to(self.device), H.to(self.device), xn.to(self.device)
        hn = H[:, -1].contiguous().view(self.batch_size * self.city_num, self.hid_dim).to(self.device)

        preds = []

        for i in range(self.forecast_len):
            emb = self.sptemp_emb(X[:, i, :, -1].long())
            x = torch.cat((xn, emb), dim=-1)
            x = x.contiguous()

            # x_in = self.fc_in(x)
            # x = torch.cat((x, x_in), dim=-1).contiguous()

            hn = self.gru_cell(x, hn)
            hn = hn.view(self.batch_size, self.city_num, self.hid_dim)
            hn = self.attn(H, hn)

            xn = self.fc_out(hn)
            hn = hn.view(self.batch_size * self.city_num, self.hid_dim)

            preds.append(xn)
        
        preds = torch.stack(preds, dim=1)
        return preds

class Seq2Seq_GNN_GRU(nn.Module):
    def __init__(self, in_dim_enc, in_dim_dec, hid_dim, city_num, num_embeddings, hist_len, forecast_len, batch_size, device, adj_mat):
        super(Seq2Seq_GNN_GRU, self).__init__()

        self.batch_size = batch_size
        self.city_num = city_num
        self.hid_dim = hid_dim
        self.out_dim = 1
        self.hist_len = hist_len

        self.Encoder = Encoder(in_dim_enc, hid_dim, city_num, num_embeddings, hist_len, batch_size, device, adj_mat)
        self.Decoder = Decoder(in_dim_dec, hid_dim, city_num, num_embeddings, hist_len, forecast_len, batch_size, device)

    def forward(self, X, y):
        
        H, xn = self.Encoder(X, y)
        xn = xn[:, -1].contiguous()
        preds = self.Decoder(X, H, xn)

        return preds