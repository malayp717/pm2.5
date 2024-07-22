import torch
import torch.nn as nn
from models.cells import GRUCell
from torch_geometric.nn import GraphConv
from torch_geometric.utils import dense_to_sparse, to_dense_adj

'''
    Encoder part for the PM2.5 History implementation
'''
class Encoder(nn.Module):
    def __init__(self, in_dim, emb_dim, hid_dim, city_num, num_embeddings, hist_len, batch_size, device, edge_indices, edge_weights):
        super(Encoder, self).__init__()
        self.emb_dim = emb_dim
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = 1

        self.city_num = city_num
        self.hist_len = hist_len
        self.batch_size = batch_size

        self.device = device

        self.edge_indices, self.edge_weights = self._process_adj_mat(edge_indices, edge_weights)

        self.spt_emb = nn.Embedding(num_embeddings, embedding_dim=self.emb_dim)
        self.conv = GraphConv(self.in_dim - 1 + self.emb_dim + 2*self.out_dim, self.hid_dim)
        self.gru_cell = GRUCell(self.in_dim -1 + self.emb_dim + 2*self.out_dim + self.hid_dim, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

    def _process_adj_mat(self, edge_indices, edge_weights):
        adj_mat = to_dense_adj(edge_index=edge_indices, edge_attr=edge_weights)
        adj_mat = adj_mat.repeat(self.batch_size, 1, 1)

        return dense_to_sparse(adj_mat)
    
    def forward(self, X, y):

        '''
            X shape: [batch_size, hist_len+forecast_len, city_num, num_features]
            y shape: [batch_size, hist_len+forecast_len, city_num, 1]
        '''

        self.edge_indices, self.edge_weights = self.edge_indices.to(self.device), self.edge_weights.to(self.device)
        X, y = X.to(self.device), y.to(self.device)

        hn = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        xn = torch.zeros(self.batch_size, self.city_num, self.out_dim).to(self.device)

        for i in range(self.hist_len):
            emb = self.spt_emb(X[:, i, :, -1].long())
            x = torch.cat((xn, y[:, i], X[:, i, :, :-1], emb), dim=-1)
            x_gcn = x.contiguous()

            x_gcn = x_gcn.view(self.batch_size * self.city_num, -1)
            x_gcn = torch.sigmoid(self.conv(x=x_gcn, edge_index=self.edge_indices, edge_weight=self.edge_weights))
            x_gcn = x_gcn.view(self.batch_size, self.city_num, -1)

            x = torch.cat((x, x_gcn), dim=-1)
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

class GraphConv_GRU(nn.Module):
    def __init__(self, in_dim_enc, in_dim_dec, emb_dim, hid_dim, city_num, num_embeddings, hist_len, forecast_len, batch_size, device,\
                 edge_indices, edge_weights):
        super(GraphConv_GRU, self).__init__()

        self.batch_size = batch_size
        self.city_num = city_num
        self.hid_dim = hid_dim
        self.out_dim = 1
        self.hist_len = hist_len

        self.Encoder = Encoder(in_dim_enc, emb_dim, hid_dim, city_num, num_embeddings, hist_len, batch_size, device,\
                               edge_indices, edge_weights)
        self.Decoder = Decoder(in_dim_dec, emb_dim, hid_dim, city_num, num_embeddings, hist_len, forecast_len, batch_size, device)

    def forward(self, X, y):
        
        hn, xn = self.Encoder(X, y)
        preds = self.Decoder(X, hn, xn)

        return preds