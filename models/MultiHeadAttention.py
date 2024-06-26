import numpy as np
import torch
import torch.nn as nn

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