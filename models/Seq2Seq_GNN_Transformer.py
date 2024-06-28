import numpy as np
import torch
import torch.nn as nn
from itertools import product
from models.Attention import MultiHeadAttention
from torch_geometric.nn import GraphConv
from torch_geometric.utils import dense_to_sparse

'''
    Encoder part for the PM2.5 History implementation
'''
class Encoder(nn.Module):
    def __init__(self, in_dim, hid_dim, city_num, hist_len, batch_size, device, adj_mat):
        super(Encoder, self).__init__()
        self.emb_dim = 8
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = 1

        self.city_num = city_num
        self.hist_len = hist_len
        self.batch_size = batch_size

        self.device = device
        self.edge_indices, self.edge_weights = self._process_adj_mat(adj_mat)

        self.pos_emb = nn.Embedding(self.hist_len, self.emb_dim)
        self.conv = GraphConv(self.in_dim + self.out_dim, self.hid_dim)
        self.fc_in = nn.Linear(self.in_dim + self.out_dim + self.emb_dim + self.hid_dim, self.hid_dim)
        self.attention = MultiHeadAttention(self.hid_dim, heads=4, mask=False)
        self.mlp = nn.Linear(self.hid_dim, self.hid_dim)

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
        X, y = X[:, :self.hist_len], y[:, :self.hist_len]

        '''
            Positional Embeddings shape: [batch_size, hist_len, city_num, 8]
        '''
        pos_emb = self.pos_emb(torch.arange(self.hist_len).to(self.device))
        pos_emb = pos_emb.repeat(self.batch_size * self.city_num, 1, 1)
        pos_emb = pos_emb.view(self.batch_size, self.city_num, self.hist_len, -1)
        pos_emb = pos_emb.transpose(1, 2).contiguous()
        word_emb = []

        '''
            Use GNN to get enhanced spatial embeddings word_emb of shape: [batch_size, hist_len, city_num, hid_dim]
        '''

        for i in range(self.hist_len):

            x = torch.cat((y[:, i], X[:, i]), dim=-1)
            x_gcn = x.contiguous()

            x_gcn = x_gcn.view(self.batch_size * self.city_num, -1)
            x_gcn = torch.sigmoid(self.conv(x=x_gcn, edge_index=self.edge_indices, edge_weight=self.edge_weights))
            x_gcn = x_gcn.view(self.batch_size, self.city_num, -1)

            word_emb.append(x_gcn)

        word_emb = torch.stack(word_emb, dim=1)
        out = torch.cat((pos_emb, word_emb, X, y), dim=-1)

        '''
            out shape: [batch_size, hist_len, city_num, in_dim + hid_dim + out_dim + emb_dim]
        '''
        out = out.transpose(1, 2).contiguous().view(self.batch_size * self.city_num, self.hist_len, -1)
        out = self.fc_in(out).view(self.batch_size, self.city_num, self.hist_len, self.hid_dim).transpose(1, 2).contiguous()

        x_attn = self.attention(out, out, out)
        out = self.mlp(x_attn)
        return out

'''
    Decoder part for the PM2.5 Forecasting implementation
'''
class Decoder(nn.Module):
    def __init__(self, in_dim, hid_dim, city_num, hist_len, forecast_len, batch_size, device):
        super(Decoder, self).__init__()
        self.emb_dim = 8
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = 1

        self.city_num = city_num
        self.hist_len = hist_len
        self.forecast_len = forecast_len
        self.batch_size = batch_size

        self.device = device

        self.pos_emb = nn.Embedding(self.forecast_len, self.emb_dim)
        '''
            no of hours in day = 24
            weekday/weekend = 2
            Therefore, total possibilities (size) = 24*2 = 48
        '''
        # self.word_emb = nn.Embedding(48, self.emb_dim)
        self.fc_in = nn.Linear(self.in_dim + self.emb_dim, 2*self.emb_dim)
        self.self_attention = MultiHeadAttention(2*self.emb_dim, heads=4, mask=True)
        self.fc_enc_dec = nn.Linear(2*self.emb_dim, self.hid_dim)
        self.enc_dec_attention = MultiHeadAttention(self.hid_dim, heads=4, mask=False)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

    def get_indices_mat(self, X):
        def cyc_emb(t):
            angle = (2.0 * np.pi * t) / 24.0
            return (np.sin(angle), np.cos(angle))
        
        t = [cyc_emb(i) for i in range(24)]
        combs = list(product(t, np.arange(0.0, 2.0, 1.0)))
        emb = [(x[0][0], x[0][1], x[1]) for x in combs]

        vector_to_index = {x: i for i, x in enumerate(emb)}

        batch_size, forecast_len, city_num, num_features = X.size()

        x = X.reshape(-1, num_features).contiguous()
        indices = []
        for row in x:
            idx = vector_to_index[tuple(row.tolist())]
            indices.append(idx)

        indices = torch.tensor(indices)
        return indices.view(batch_size, forecast_len, city_num)

    def forward(self, X, enc_out):

        '''
            X shape: [batch_size, hist_len+forecast_len, city_num, num_features]
            new X shape: [batch_size, forecast_len, city_num, 3], since only last 3 features available at forecasting time
        '''
        X = X.to(self.device)
        X = X[:, self.hist_len:, :, -3:]

        '''
            Positional Embeddings shape: [batch_size, forecast_len, city_num, 8]
        '''
        pos_emb = self.pos_emb(torch.arange(self.forecast_len).to(self.device))
        pos_emb = pos_emb.repeat(self.batch_size * self.city_num, 1, 1)
        pos_emb = pos_emb.view(self.batch_size, self.city_num, self.hist_len, -1)
        pos_emb = pos_emb.transpose(1, 2).contiguous()
        
        '''
            Word Embeddings shape: [batch_size, forecast_len, city_num, 8]
            Concat: [batch_size, forecast_len, city_num, 16 (=8+8)]
        '''
        # word_emb = self.get_indices_mat(X)
        # word_emb = self.word_emb(word_emb)

        # out = torch.cat((pos_emb, word_emb), dim=-1)

        '''
            out shape: [batch_size, forecast_len, city_num, 11 (=8+3)]
        '''
        out = torch.cat((X, pos_emb), dim=-1)
        out = out.transpose(1, 2).contiguous().view(self.batch_size * self.city_num, self.forecast_len, -1)
        out = self.fc_in(out).view(self.batch_size, self.city_num, self.forecast_len, -1).transpose(1, 2).contiguous()

        x = self.self_attention(out, out, out)
        # x shape: [batch_size, forecast_len, city_num, 2*emb_dim]
        x = x.transpose(1, 2).contiguous().view(self.batch_size * self.city_num, self.forecast_len, -1)
        x = self.fc_enc_dec(x)
        x = x.view(self.batch_size, self.city_num, self.forecast_len, self.hid_dim).transpose(1, 2).contiguous()
        # x shape: [batch_size, forecast_len, city_num, hid_dim]
        x = self.enc_dec_attention(x, enc_out, enc_out)
        # x shape: [batch_size, forecast_len, city_num, hid_dim]

        preds = []
        for i in range(self.forecast_len):
            xn = x[:, i]
            xn = self.fc_out(xn)

            preds.append(xn)

        preds = torch.stack(preds, dim=1)
        return preds

class Seq2Seq_GNN_Transformer(nn.Module):
    def __init__(self, in_dim_enc, in_dim_dec, hid_dim, city_num, hist_len, forecast_len, batch_size, device, adj_mat):
        super(Seq2Seq_GNN_Transformer, self).__init__()

        self.batch_size = batch_size
        self.city_num = city_num
        self.hid_dim = hid_dim
        self.out_dim = 1
        self.hist_len = hist_len

        self.Encoder = Encoder(in_dim_enc, hid_dim, city_num, hist_len, batch_size, device, adj_mat)
        self.Decoder = Decoder(in_dim_dec, hid_dim, city_num, hist_len, forecast_len, batch_size, device)

    def forward(self, X, y):
        
        enc_out = self.Encoder(X, y)
        preds = self.Decoder(X, enc_out)
        return preds