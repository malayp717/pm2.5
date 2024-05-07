import numpy as np
import torch
import torch.nn as nn
from models.cells import GRUCell

class GRU(nn.Module):
    def __init__(self, in_dim, hid_dim, city_num, hist_window, forecast_window, batch_size, device):
        super(GRU, self).__init__()
        self.device = device
        self.hist_window = hist_window
        self.forecast_window = forecast_window
        # self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = 1
        self.city_num = city_num
        self.batch_size = batch_size

        self.fc_in = nn.Linear(in_dim+1, hid_dim)
        self.gru_cell = GRUCell(hid_dim, hid_dim)
        self.fc_out = nn.Linear(hid_dim, self.out_dim)

    def forward(self, feature, pm25_hist):
        '''
            feature shape: (batch_size, forecast_window, number of features)
            pm25_hist shape: (batch_size, T-forecast_window), where T are total number of timestamps in the dataset 
        '''
        feature, pm25_hist = feature.to(self.device), pm25_hist.to(self.device)
        pm25_pred = []

        h0 = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        hn = h0
        xn = pm25_hist[:, -1]

        for i in range(self.forecast_window):
            x = torch.cat((xn, feature[:, self.hist_window+i]), dim=-1)
            x = self.fc_in(x)
            
            hn = self.gru_cell(x, hn)
            xn = hn.view(self.batch_size, self.city_num, self.hid_dim)
            xn = self.fc_out(xn)
            pm25_pred.append(xn)

        pm25_pred = torch.stack(pm25_pred, dim=1)
        return pm25_pred