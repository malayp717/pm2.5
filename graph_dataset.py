import numpy as np
import time
import torch
import torch.nn as nn
# from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GraphConv, ChebConv
from torch.optim import lr_scheduler
# from torchsummary import summary
from constants import *
from utils import *
from dataset.SpatioTemporalDataset import SpatioTemporalDataset
import warnings
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device):
        super(GNNLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device

        self.embedding = nn.Linear(self.input_dim, self.hidden_dim)
        self.lstm_cell_1 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.lstm_cell_2 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        # self.gru_cell_1 = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        # self.gru_cell_2 = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        # self.conv = GCNConv(self.hidden_dim*2, self.hidden_dim*2, add_self_loops=True)
        # self.conv = GATConv(self.hidden_dim*2, self.hidden_dim, add_self_loops=True)
        # self.conv = SAGEConv(self.hidden_dim*2, self.hidden_dim)
        # self.conv = GraphConv(self.hidden_dim*2, self.hidden_dim, aggr='mean')
        # self.conv_1 = GCNConv(self.hidden_dim, self.hidden_dim, add_self_loops=True)
        # self.conv_2 = GCNConv(self.hidden_dim, self.hidden_dim, add_self_loops=True)
        self.conv_1 = ChebConv(self.hidden_dim, self.hidden_dim, K=2)
        self.conv_2 = ChebConv(self.hidden_dim, self.hidden_dim, K=2)
        self.dropout = nn.Dropout(p=0.1)
        # self.activation = nn.SiLU()
        self.fc = nn.Linear(self.hidden_dim*2, self.output_dim)

    def forward(self, x, edge_index):
        h_1, c_1 = torch.zeros(x.size(0), self.hidden_dim), torch.zeros(x.size(0), self.hidden_dim)
        h_1, c_1 = h_1.to(self.device), c_1.to(self.device)

        h_2, c_2 = torch.zeros(x.size(0), self.hidden_dim), torch.zeros(x.size(0), self.hidden_dim)
        h_2, c_2 = h_2.to(self.device), c_2.to(self.device)

        for t in range(x.size(1)):
            input_features_1 = self.dropout(self.embedding(x[:, t, :]))
            input_features_2 = self.dropout(self.embedding(x[:, x.size(1)-t-1, :]))

            h_1, c_1 = self.lstm_cell_1(input_features_1, (h_1, c_1))
            h_2, c_2 = self.lstm_cell_2(input_features_2, (h_2, c_2))

            # h_1 = self.gru_cell_1(input_features_1, h_1)
            # h_2 = self.gru_cell_2(input_features_2, h_2)

            # h = torch.cat((h_1, h_2), dim=1)
            # h = self.conv(h, edge_index)

            # h_1, h_2 = h[:, :self.hidden_dim], h[:, self.hidden_dim:]

            h_1 = self.conv_1(h_1, edge_index)
            h_2 = self.conv_2(h_2, edge_index)

            # print(h.size(), h_1.size(), h_2.size())

        out = self.dropout(self.fc(torch.cat((h_1, h_2), dim=1)))
        return out
    
def summary(model):
    '''
        Counts the total number of trainable parameters in a PyTorch model.
        Args:
        model: A PyTorch model.
        Returns:
        The total number of trainable parameters in the model.
    '''
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    return total_params


def l1_regularizer(model, device, lambda_=0.01):
    '''
    Calculates L1 regularization loss for a model.

    Args:
        model: PyTorch model object.
        lambda_: Regularization weight (controls strength of L1 penalty).

    Returns:
        L1 regularization loss (tensor).
    '''
    l1_reg = torch.tensor(0.).to(device)
    for param in model.parameters():
        l1_reg += torch.abs(param).sum()
    return lambda_ * l1_reg

if __name__ == '__main__':

    FW, WS, LR, NUM_EPOCHS, BATCH_SIZE, HIDDEN_DIM = 6, 300, 5e-3, 5, 4, 64
    locs_file = f'{data_bihar}/locs.txt'
    train_fp = f'{gnn_data_bihar}/bihar_fw:{FW}_ws:{WS}_train_gnn.pkl'
    val_fp = f'{gnn_data_bihar}/bihar_fw:{FW}_ws:{WS}_validation_gnn.pkl'
    test_fp = f'{gnn_data_bihar}/bihar_fw:{FW}_ws:{WS}_test_gnn.pkl'

    locs = load_locs_as_tuples(locs_file)
    train_data = SpatioTemporalDataset(train_fp, locs, HIDDEN_DIM) 
    val_data = SpatioTemporalDataset(val_fp, locs, HIDDEN_DIM)
    test_data = SpatioTemporalDataset(test_fp, locs, HIDDEN_DIM)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)
    # print(test_data[0])

    input_dim, hidden_dim, output_dim = train_data.get_input_dimensions()
    # output_dim = 1
    print(input_dim, hidden_dim, output_dim)

    model = GNNLSTM(input_dim, hidden_dim, output_dim, device)
    print(f'Total parameters in model: {summary(model)}')
    model.to(device)

    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train_losses, val_losses = [], []
    print(f"---------\t Training started lr={LR},  hidden_size={hidden_dim} \t---------")
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        model.train()
        for data in train_loader:
            # x, y, edge_index = data.x.to(device), data.y[:, -1, :].to(device), data.edge_index.to(device)
            x, y, edge_index = data
            x, y, edge_index = x.to(device), y.to(device), edge_index.to(device)
            # x, y, edge_index = data.x.to(device), data.y[:, -1].to(device), data.edge_index.to(device)

            preds = model(x, edge_index)
            
            # train_loss = torch.sqrt(criterion(y, preds))
            train_loss = criterion(y, preds) + l1_regularizer(model, device)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_losses.append(train_loss.item())

        model.eval()
        with torch.no_grad():
            for data in val_loader:
                # x, y, edge_index = data.x.to(device), data.y[:, -1, :].to(device), data.edge_index.to(device)
                x, y, edge_index = data
                x, y, edge_index = x.to(device), y.to(device), edge_index.to(device)
                # x, y, edge_index = data.x.to(device), data.y[:, -1].to(device), data.edge_index.to(device)
                preds = model(x, edge_index)
                
                # val_loss = torch.sqrt(criterion(y, preds))
                val_loss = criterion(y, preds) + l1_regularizer(model, device)
                val_losses.append(train_loss.item())

        scheduler.step()

        print(f'Epoch: {epoch+1}/{NUM_EPOCHS}, lr: {scheduler.get_last_lr()}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, \
        time_taken: {(time.time()-start_time)/60:.2f} mins')

    test_loss = None

    preds_train, preds_val, preds_test = [], [], []
    y_train, y_val, y_test = [], [], []

    model.eval()
    with torch.no_grad():
        for data in train_loader:
            # x, y, edge_index = data.x.to(device), data.y[:, -1, :].to(device), data.edge_index.to(device)
            x, y, edge_index = data
            x, y, edge_index = x.to(device), y.to(device), edge_index.to(device)
            # x, y, edge_index = data.x.to(device), data.y[:, -1].to(device), data.edge_index.to(device)

            preds = model(x, edge_index)
            torch.clamp_min_(preds, 0)

            y_train.extend(y.cpu().tolist())
            preds_train.extend(preds.cpu().tolist())

        with torch.no_grad():
            for data in val_loader:
                # x, y, edge_index = data.x.to(device), data.y[:, -1, :].to(device), data.edge_index.to(device)
                x, y, edge_index = data
                x, y, edge_index = x.to(device), y.to(device), edge_index.to(device)
                # x, y, edge_index = data.x.to(device), data.y[:, -1].to(device), data.edge_index.to(device)
                preds = model(x, edge_index)
                torch.clamp_min_(preds, 0)

                y_val.extend(y.cpu().tolist())
                preds_val.extend(preds.cpu().tolist())

        for data in test_loader:
            # x, y, edge_index = data.x.to(device), data.y[:, -1, :].to(device), data.edge_index.to(device)
            x, y, edge_index = data
            x, y, edge_index = x.to(device), y.to(device), edge_index.to(device)
            # x, y, edge_index = data.x.to(device), data.y[:, -1].to(device), data.edge_index.to(device)
            preds = model(x, edge_index)
            
            # test_loss = torch.sqrt(criterion(y, preds))
            test_loss = criterion(y, preds) + l1_regularizer(model, device)
            torch.clamp_min_(preds, 0)

            y_test.extend(y.cpu().tolist())
            preds_test.extend(preds.cpu().tolist())
        
    print(f'Test Loss: {test_loss.item():.4f}')

    y_train, preds_train, y_val, preds_val, y_test, preds_test = np.array(y_train), np.array(preds_train), np.array(y_val),\
                                                                    np.array(preds_val), np.array(y_test), np.array(preds_test)
    y_train, preds_train, y_val, preds_val, y_test, preds_test = y_train.reshape(-1), preds_train.reshape(-1), y_val.reshape(-1),\
                                                                    preds_val.reshape(-1), y_test.reshape(-1), preds_test.reshape(-1)
    
    print(f"Train Stats (RMSE, R_squared, p_value, R_squared_pearson, p_value_pearson)")
    print(eval_stat(preds_train, y_train))

    print(f"\nVal Stats (RMSE, R_squared, p_value, R_squared_pearson, p_value_pearson)")
    print(eval_stat(preds_val, y_val))

    print(f"\nTest Stats (RMSE, R_squared, p_value, R_squared_pearson, p_value_pearson)")
    print(eval_stat(preds_test, y_test))

    print(preds_train[0:24], y_train[0:24])
    print(preds_val[0:24], y_val[0:24])
    print(preds_test[0:24], y_test[0:24])