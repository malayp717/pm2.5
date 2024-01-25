import torch
import torch.nn as nn

class InvalidRNNTypeException(Exception):
    "Raised when RNN variant passed as an argument is other than RNN, LSTM or GRU"
    pass

class RNN(nn.Module):
    def __init__(self, _type, input_dim, layer_dim, hidden_dim, bidirectional, device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.bidirectional = bidirectional
        self.device = device
        self.type = _type
        if self.type == 'RNN':
            self.series = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu',\
                                 bidirectional=self.bidirectional)
        elif self.type == 'LSTM':
            self.series = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=self.bidirectional)
        elif self.type == 'GRU':
            self.series = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=self.bidirectional)
        else:
            raise InvalidRNNTypeException
        if self.bidirectional == True:
            self.fc = nn.Linear(hidden_dim*2, 1)
        else:
            self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        
        out, _ = self.series(x)

        outs = []    # save all predictions
        
        for time_step in range(out.size(1)):    # calculate output for each time step
            outs.append(self.fc(out[:, time_step, :]))

        outs =  torch.stack(outs, dim=1).to(self.device)

        return outs