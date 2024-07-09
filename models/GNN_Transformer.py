import numpy as np
import torch
import torch.nn as nn
from itertools import product
from metpy.units import units
from metpy.calc import wind_direction, wind_speed
from models.Attention import MultiHeadAttention
from torch_geometric.nn import TransformerConv
# from torch_geometric.utils import dense_to_sparse

'''
    Encoder part for the PM2.5 History implementation
'''
class Encoder(nn.Module):
    def __init__(self, in_dim, emb_dim, hid_dim, city_num, num_embeddings, hist_len, batch_size, device,\
                    edge_indices, edge_attr, u10_mean, u10_std, v10_mean, v10_std, edge_dim):
        super(Encoder, self).__init__()
        self.emb_dim = emb_dim
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = 1

        self.city_num = city_num
        self.hist_len = hist_len
        self.batch_size = batch_size

        self.device = device
        self.edge_indices, self.edge_attr = edge_indices, edge_attr
        self.u10_mean, self.u10_std = u10_mean, u10_std
        self.v10_mean, self.v10_std = v10_mean, v10_std

        self.pos_emb = nn.Embedding(self.hist_len, self.emb_dim)
        self.spt_emb = nn.Embedding(num_embeddings, embedding_dim=self.emb_dim)
        self.conv = TransformerConv(self.in_dim - 1 + self.emb_dim + self.out_dim, self.hid_dim, edge_dim=edge_dim,\
                                    dropout=0.5)
        self.fc_in = nn.Linear(self.in_dim + self.out_dim + self.emb_dim + self.hid_dim, self.hid_dim)
        self.attention = MultiHeadAttention(self.hid_dim, heads=4, mask=False)
        self.mlp = nn.Linear(self.hid_dim, self.hid_dim)

    def _compute_edge_attr(self, X):
        ''' 
            edge_indices: [2, E]
            edge_attr: [E, D]
            X shape: [batch_size, num_locs, num_features]                   NOTE: X doesn't have embedding column in it
            last column: v10
            second last column: u10
        '''
        _, D = self.edge_attr.size()
        edge_attr = self.edge_attr.view(1, -1, D).repeat(self.batch_size, 1, 1)
        edge_attr = edge_attr.transpose(1, 0).contiguous()

        '''
            u10 shape: [batch_size, num_locs]
            v10 shape: [batch_size, num_locs]
        '''
        u10 = X[:, :, -2] * self.u10_std + self.u10_mean
        v10 = X[:, :, -1] * self.v10_std + self.v10_mean

        u10, v10 = u10.cpu().detach().numpy(), v10.cpu().detach().numpy()
        u10, v10 = u10.reshape(-1) * units('m/s'), v10.reshape(-1) * units('m/s')

        speed = wind_speed(u10, v10).magnitude
        direction = wind_direction(u10, v10).to('radians').magnitude

        speed, direction = speed.reshape(self.batch_size, self.city_num), direction.reshape(self.batch_size, self.city_num)
        speed, direction = torch.tensor(speed, dtype=torch.float32), torch.tensor(direction, dtype=torch.float32)

        wind_attr = torch.stack([speed, direction], axis=2).to(self.device)

        EDGE_ATTR = []

        for i in range(self.edge_indices.size(1)):
            src = self.edge_indices[0, i]
            # attr: [distance, src_to_dst_angle, wind_speed, wind_direction]
            attr = torch.concat([edge_attr[src, :, :], wind_attr[:, src, :]], axis=-1)

            theta = torch.abs(attr[:, 1] - attr[:, 3])
            advection_coeff = torch.relu(3 * attr[:, 2] * torch.cos(theta) / attr[:, 0]).view(-1, 1)

            attr = torch.concat([attr, advection_coeff], axis=-1)
            EDGE_ATTR.append(attr)

        # EDGE_ATTR shape: [E, batch_size, D]
        EDGE_ATTR = torch.stack(EDGE_ATTR, axis=0)
        _, _, D = EDGE_ATTR.size()
        EDGE_ATTR = EDGE_ATTR.view(-1, D)

        EDGE_INDICES = self.edge_indices.view(2, 1, -1).repeat(1, self.batch_size, 1)\
                        + torch.arange(self.batch_size).view(1, -1, 1).to(self.device) * self.city_num
        EDGE_INDICES = EDGE_INDICES.view(2, -1)

        return EDGE_INDICES, EDGE_ATTR
    
    def forward(self, X, y):

        '''
            X shape: [batch_size, hist_len+forecast_len, city_num, num_features]
            y shape: [batch_size, hist_len+forecast_len, city_num, 1]
        '''

        self.edge_indices, self.edge_attr = self.edge_indices.to(self.device), self.edge_attr.to(self.device)
        X, y = X.to(self.device), y.to(self.device)
        X, y = X[:, :self.hist_len], y[:, :self.hist_len]

        '''
            Positional Embeddings shape: [batch_size, hist_len, city_num, emb_dim]
        '''
        pos_emb = self.pos_emb(torch.arange(self.hist_len).to(self.device))
        pos_emb = pos_emb.repeat(self.batch_size * self.city_num, 1, 1)
        pos_emb = pos_emb.view(self.batch_size, self.city_num, self.hist_len, -1)
        pos_emb = pos_emb.transpose(1, 2).contiguous()

        '''
            Embeddings for Spatio Temporal Features: [batch_size, hist_len, city_num, emb_dim]
        '''
        spt_emb = self.spt_emb(X[:, :, :, -1].long())
        emb = pos_emb + spt_emb
        word_emb = []

        '''
            Use GNN to get enhanced spatial embeddings word_emb of shape: [batch_size, hist_len, city_num, hid_dim]
        '''

        for i in range(self.hist_len):
            
            x = torch.cat((emb[:, i], y[:, i], X[:, i, :, :-1]), dim=-1)
            x_gcn = x.contiguous()

            edge_indices, edge_attr = self._compute_edge_attr(X[:, i, :, : -1])

            x_gcn = x_gcn.view(self.batch_size * self.city_num, -1)
            x_gcn = torch.sigmoid(self.conv(x=x_gcn, edge_index=edge_indices, edge_attr=edge_attr))
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
    def __init__(self, in_dim, emb_dim, hid_dim, city_num, num_embeddings, hist_len, forecast_len, batch_size, device):
        super(Decoder, self).__init__()
        self.emb_dim = emb_dim
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = 1

        self.city_num = city_num
        self.hist_len = hist_len
        self.forecast_len = forecast_len
        self.batch_size = batch_size

        self.device = device

        self.pos_emb = nn.Embedding(self.forecast_len, self.emb_dim)
        self.spt_emb = nn.Embedding(num_embeddings, embedding_dim=self.emb_dim)
        self.fc_in = nn.Linear(self.emb_dim, 2*self.emb_dim)
        self.self_attention = MultiHeadAttention(2*self.emb_dim, heads=4, mask=True)
        self.fc_enc_dec = nn.Linear(2*self.emb_dim, self.hid_dim)
        self.enc_dec_attention = MultiHeadAttention(self.hid_dim, heads=4, mask=False)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

    def forward(self, X, enc_out):

        '''
            X shape: [batch_size, hist_len+forecast_len, city_num, num_features]
            emb shape: [batch_size, forecast_len, city_num, emb_dim], since only last feature is available at forecasting time
        '''
        X = X.to(self.device)
        X = X[:, self.hist_len:, :, -1]

        '''
            Positional Embeddings shape: [batch_size, forecast_len, city_num, emb_dim]
        '''
        pos_emb = self.pos_emb(torch.arange(self.forecast_len).to(self.device))
        pos_emb = pos_emb.repeat(self.batch_size * self.city_num, 1, 1)
        pos_emb = pos_emb.view(self.batch_size, self.city_num, self.hist_len, -1)
        pos_emb = pos_emb.transpose(1, 2).contiguous()
        
        '''
            Word Embeddings shape: [batch_size, forecast_len, city_num, emb_dim]
            out shape: [batch_size, forecast_len, city_num, emb_dim]
        '''
        spt_emb = self.spt_emb(X.long())

        out = pos_emb + spt_emb
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

class GNN_Transformer(nn.Module):
    def __init__(self, in_dim_enc, in_dim_dec, emb_dim, hid_dim, city_num, num_embeddings, hist_len, forecast_len, batch_size, device,\
                 edge_indices, edge_attr, u10_mean, u10_std, v10_mean, v10_std, edge_dim):
        super(GNN_Transformer, self).__init__()

        self.batch_size = batch_size
        self.city_num = city_num
        self.hid_dim = hid_dim
        self.out_dim = 1
        self.hist_len = hist_len

        self.Encoder = Encoder(in_dim_enc, emb_dim, hid_dim, city_num, num_embeddings, hist_len, batch_size, device,\
                               edge_indices, edge_attr, u10_mean, u10_std, v10_mean, v10_std, edge_dim)
        self.Decoder = Decoder(in_dim_dec, emb_dim, hid_dim, city_num, num_embeddings, hist_len, forecast_len, batch_size, device)

    def forward(self, X, y):
        
        enc_out = self.Encoder(X, y)
        preds = self.Decoder(X, enc_out)
        return preds