import numpy as np
import torch
import torch.nn as nn
from models.cells import GRUCell

'''
    Encoder part for the PM2.5 History implementation
'''
class Encoder(nn.Module):
    def __init__(self, in_dim, emb_dim, hid_dim, city_num, num_embeddings, hist_len, batch_size, device):
        super(Encoder, self).__init__()
        self.emb_dim = emb_dim
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = 1

        self.city_num = city_num
        self.hist_len = hist_len
        self.batch_size = batch_size

        self.device = device

        self.spt_emb = nn.Embedding(num_embeddings, embedding_dim=self.emb_dim)
        self.gru_cell = GRUCell(self.in_dim - 1 + self.emb_dim + 2*self.out_dim, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)
    
    def forward(self, X, y):

        '''
            X shape: [batch_size, hist_len+forecast_len, city_num, num_features]
            y shape: [batch_size, hist_len+forecast_len, city_num, 1]
        '''
        X, y = X.to(self.device), y.to(self.device)

        hn = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        xn = torch.zeros(self.batch_size, self.city_num, self.out_dim).to(self.device)

        for i in range(self.hist_len):
            emb = self.spt_emb(X[:, i, :, -1].long())
            x = torch.cat((xn, y[:, i], X[:, i, :, :-1], emb), dim=-1)
            x = x.contiguous()

            hn = self.gru_cell(x, hn)

            xn = hn.view(self.batch_size, self.city_num, self.hid_dim)
            xn = self.fc_out(xn)

        return hn, xn

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

        self.spt_emb = nn.Embedding(num_embeddings, embedding_dim=self.emb_dim)
        self.gru_cell = GRUCell(self.emb_dim + self.out_dim, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

    def forward(self, X, hn, xn):

        ''' 
            X shape: [batch_size, hist_len+forecast_len, city_num, num_features]
            H shape: [batch_size, hist_len, city_num, hid_dim]
            xn shape: [batch_size, city_num, out_dim]
        '''
        X, hn, xn = X.to(self.device), hn.to(self.device), xn.to(self.device)
        preds = []

        for i in range(self.forecast_len):
            emb = self.spt_emb(X[:, i, :, -1].long())
            x = torch.cat((xn, emb), dim=-1)
            x = x.contiguous()

            hn = self.gru_cell(x, hn)
            xn = hn.view(self.batch_size, self.city_num, self.hid_dim)

            xn = self.fc_out(xn)
            preds.append(xn)
        
        preds = torch.stack(preds, dim=1)
        return preds

class GRU(nn.Module):
    def __init__(self, in_dim_enc, in_dim_dec, emb_dim, hid_dim, city_num, num_embeddings, hist_len, forecast_len, batch_size, device):
        super(GRU, self).__init__()

        self.batch_size = batch_size
        self.city_num = city_num
        self.hid_dim = hid_dim
        self.out_dim = 1
        self.hist_len = hist_len

        self.Encoder = Encoder(in_dim_enc, emb_dim, hid_dim, city_num, num_embeddings, hist_len, batch_size, device)
        self.Decoder = Decoder(in_dim_dec, emb_dim, hid_dim, city_num, num_embeddings, hist_len, forecast_len, batch_size, device)

    def forward(self, X, y):
        
        hn, xn = self.Encoder(X, y)
        preds = self.Decoder(X, hn, xn)

        return preds