import numpy as np
import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, forecast_window, device):
        super(GRU, self).__init__()
        self.device = device
        self.forecast_window = forecast_window
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # self.gru_hist = nn.GRU(1, hidden_dim, batch_first=True, bidirectional=True)
        # self.fc = nn.Linear(input_dim+2*hidden_dim, hidden_dim)
        self.fc_hist = nn.Linear(1, hidden_dim)
        self.gru_hist_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc_in = nn.Linear(hidden_dim+input_dim, hidden_dim)
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, feature, pm25_hist):
        '''
            feature shape: (batch_size, forecast_window, number of features)
            pm25_hist shape: (batch_size, T-forecast_window), where T are total number of timestamps in the dataset 
        '''
        feature, pm25_hist = feature.to(self.device), pm25_hist.to(self.device)
        pm25_hist = torch.unsqueeze(pm25_hist, -1)
        pm25_pred = []

        # xn, _ = self.gru_hist(pm25_hist)
        # xn = xn[:, -1, :]
        h_hist = torch.zeros(feature.size(0), self.hidden_dim).to(self.device)

        for i in range(pm25_hist.size(1)):
            x = self.fc_hist(pm25_hist[:, i])
            h_hist = self.gru_hist_cell(x, h_hist)

        # print(xn.size())
        h0 = h_hist
        hn = torch.zeros(h0.size(0), self.hidden_dim).to(self.device)

        for i in range(self.forecast_window):
            x = torch.cat((h0, feature[:, i]), dim=-1) 
            x = self.fc_in(x)
            hn = self.gru_cell(x, hn)
            h0 = hn

            # print(x.size(), hn.size())
            xn = self.fc_out(hn)
            pm25_pred.append(xn)

        pm25_pred = torch.stack(pm25_pred, dim=1).squeeze()
        return pm25_pred