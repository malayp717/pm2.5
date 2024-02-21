import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        x, y = sample['meteo'], sample['pm25']
        return torch.Tensor(x), torch.Tensor(y)

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, out_dim, bidirectional):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        # self.gelu = nn.GELU()
        self.fc = nn.Linear(hidden_dim*2, out_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        # out = self.gelu(out)
        out = out[:, -1, :]

        out = self.fc(out)
        # out = LOWER_BOUND + (out * (UPPER_BOUND - LOWER_BOUND))
        # out = torch.clamp(out, LOWER_BOUND, UPPER_BOUND)
        return out
    
class FrobeniusNorm(nn.Module):
    def __init__(self):
        super(FrobeniusNorm, self).__init__()
    
    def forward(self, y, y_pred):
        return torch.sqrt(torch.mean(torch.norm(y_pred-y, dim=1)))