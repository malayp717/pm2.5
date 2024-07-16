import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

'''
    Multiheaded Self Attention
'''
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, heads, mask=False):
        super(MultiHeadAttention, self).__init__()
        
        assert emb_dim % heads == 0
        self.emb_dim, self.heads, self.mask = emb_dim, heads, mask
        self.scale_factor = 1/np.sqrt(self.emb_dim // self.heads)

        self.w_k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_v = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_concat = nn.Linear(emb_dim, emb_dim)

    def dot_product_attention(self, q, k, v):

        '''
            q, k, v shape: [batch_size * city_num, num_heads, hist_len, red_dim]
            red_dim * num_heads = self.emb_dim

            NOTE: In the below code, it is NOT the true batch_size, it is actually equal to batch_size * city_num
        '''
        batch_size, num_heads, hist_len, red_dim = q.size()

        q = q.view(batch_size * num_heads, hist_len, red_dim)
        k = k.view(batch_size * num_heads, hist_len, red_dim)
        v = v.view(batch_size * num_heads, hist_len, red_dim)

        dot = torch.bmm(q, k.transpose(1, 2))
        dot = dot * self.scale_factor

        assert dot.size() == (batch_size * num_heads, hist_len, hist_len)

        if self.mask:
            indices = torch.triu_indices(hist_len, hist_len, offset=1)
            dot[:, indices[0], indices[1]] = float('-inf')

        dot = torch.softmax(dot, dim=2)
        out = torch.bmm(dot, v).view(batch_size, num_heads, hist_len, red_dim)

        return out

    def forward(self, values, queries, keys):

        batch_size, hist_len, city_num, emb_dim = values.size()
        assert emb_dim == self.emb_dim

        # Reduced dimension of each head
        red_dim = self.emb_dim // self.heads
        assert red_dim * self.heads == emb_dim

        '''
            values, keys, queries shape: [batch_size, hist_len, city_num, emb_dim]
        '''
        values = values.transpose(1, 2).contiguous().view(batch_size * city_num, hist_len, emb_dim)
        queries = queries.transpose(1, 2).contiguous().view(batch_size * city_num, hist_len, emb_dim)
        keys = keys.transpose(1, 2).contiguous().view(batch_size * city_num, hist_len, emb_dim)

        # q, k, v shape: [batch_size * city_num, hist_len, emb_dim]
        q, k, v = self.w_q(queries), self.w_k(keys), self.w_v(values)

        # q, k, v shape: [batch_size * city_num, num_heads, hist_len, red_dim]
        q = q.view(batch_size * city_num, hist_len, self.heads, red_dim).transpose(1, 2).contiguous()
        k = k.view(batch_size * city_num, hist_len, self.heads, red_dim).transpose(1, 2).contiguous()
        v = v.view(batch_size * city_num, hist_len, self.heads, red_dim).transpose(1, 2).contiguous()
        
        # out shape: [batch_size * city_num, num_heads, hist_len, red_dim]
        out = self.dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(batch_size * city_num, hist_len, self.heads * red_dim)
        assert out.size() == (batch_size * city_num, hist_len, self.emb_dim)

        out = self.w_concat(out)
        out = out.view(batch_size, city_num, hist_len, emb_dim).transpose(1, 2).contiguous()

        return out

class LuongAttention(nn.Module):
    def __init__(self, hid_dim):
        super(LuongAttention, self).__init__()
        self.hid_dim = hid_dim

        self.W = nn.Linear(self.hid_dim, self.hid_dim)

    def forward(self, H, x):
        '''
            H shape: [batch_size, hist_len, num_locs, hid_dim]
            x shape: [batch_size, num_locs, hid_dim]
        '''
        batch_size, hist_len, num_locs, hid_dim = H.size()

        # H shape: [batch_size * num_locs, hist_len, hid_dim]
        H = H.transpose(1, 2).contiguous().view(batch_size * num_locs, hist_len, hid_dim)
        # x shape: [batch_size * num_locs, hist_len, 1, hid_dim]
        x = x.view(batch_size * num_locs, 1, hid_dim)

        # out shape: [batch_size * num_locs, hist_len, hid_dim]
        out = self.W(H)
        # scores shape: [batch_size * num_locs, 1, hist_len]
        scores = torch.bmm(x, out.transpose(1, 2))
        # weights shape: [batch_size * num_locs, hist_len, 1]
        weights = torch.softmax(scores, dim=2).view(batch_size * num_locs, hist_len, 1)

        # out shape: [batch_size * num_locs, hist_len, hid_dim]
        out = weights * H
        # out shape: [batch_size * num_locs, hid_dim]
        out = torch.sum(out, dim=1)

        return out.view(batch_size, num_locs, hid_dim)
    
class GraphAttention(nn.Module):
    def __init__(self, batch_size, city_num, hid_dim, edge_indices, device):
        super(GraphAttention, self).__init__()
        self.batch_size = batch_size
        self.city_num = city_num
        self.hid_dim = hid_dim
        self.device = device

        self.edge_indices = edge_indices.view(2, 1, -1).repeat(1, batch_size, 1)\
                        + torch.arange(batch_size).view(1, -1, 1) * city_num
        self.edge_indices = self.edge_indices.view(2, -1)
        
        self.conv = GCNConv(self.hid_dim, self.hid_dim)
        # self.fc = nn.Linear(self.hid_dim, self.hid_dim)

    def forward(self, H, x):

        self.edge_indices = self.edge_indices.to(self.device)

        '''
            H shape: [batch_size, hist_len, num_locs, hid_dim]
            x shape: [batch_size, num_locs, hid_dim]
        '''
        batch_size, hist_len, num_locs, hid_dim = H.size()

        assert batch_size == self.batch_size
        assert num_locs == self.city_num
        assert hid_dim == self.hid_dim

        # H shape: [batch_size * num_locs, hist_len, hid_dim]
        H = H.transpose(1, 2).contiguous().view(batch_size * num_locs, hist_len, hid_dim)
        # x shape: [batch_size * num_locs, hist_len, 1, hid_dim]
        x = x.view(batch_size * num_locs, 1, hid_dim)

        H_gcn = []

        for i in range(hist_len):
            # h shape: [batch_size * num_locs, hid_dim]
            h = H[:, i]
            # x_gcn shape: [batch_size * num_locs, hid_dim]
            h_gcn = torch.sigmoid(self.conv(x=h, edge_index=self.edge_indices))
            H_gcn.append(h_gcn)

        # H_gcn shape: [batch_size * num_locs, hist_len, hid_dim]
        H_gcn = torch.stack(H_gcn, dim=1)
        # scores shape: [batch_size * num_locs, 1, hist_len]
        scores = torch.bmm(x, H_gcn.transpose(1, 2))
        # weights shape: [batch_size * num_locs, hist_len, 1]
        weights = torch.softmax(scores, dim=2).view(batch_size * num_locs, hist_len, 1)

        # out shape: [batch_size * num_locs, hist_len, hid_dim]
        out = weights * H_gcn
        # out shape: [batch_size * num_locs, hid_dim]
        out = torch.sum(out, dim=1)

        return out.view(batch_size, num_locs, hid_dim)
