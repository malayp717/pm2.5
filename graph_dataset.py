import numpy as np
import time
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch.optim import lr_scheduler
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
        # self.lstm_cell = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.gru_cell = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.conv = GCNConv(self.hidden_dim, self.hidden_dim, add_self_loops=True)
        # self.conv = GATConv(self.hidden_dim, self.hidden_dim, add_self_loops=True)
        # self.conv = SAGEConv(self.hidden_dim, self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x, edge_index):
        h, c = torch.zeros(x.size(0), self.hidden_dim), torch.zeros(x.size(0), self.hidden_dim)
        h, c = h.to(self.device), c.to(self.device)

        for t in range(x.size(1)):
            input_features = self.embedding(x[:, t, :])
            # h, c = self.lstm_cell(input_features, (h, c))
            h = self.gru_cell(input_features, h)
            h = self.conv(h, edge_index)

        out = self.fc(h)
        return out

if __name__ == '__main__':
    
    # train_locs = load_locs_as_tuples(f'{data_bihar}/train_locations.txt')
    # val_locs = load_locs_as_tuples(f'{data_bihar}/val_locations.txt')
    # test_locs = load_locs_as_tuples(f'{data_bihar}/test_locations.txt')

    DIST_THRESH, FW, WS, LR, NUM_EPOCHS, BATCH_SIZE, HIDDEN_DIM = 100, 12, 600, 1e-3, 10, 4, 64
    locs_file = f'{data_bihar}/locs.txt'
    train_fp = f'{data_bihar}/bihar_fw:{FW}_ws:{WS}_train_gnn.pkl'
    val_fp = f'{data_bihar}/bihar_fw:{FW}_ws:{WS}_validation_gnn.pkl'
    test_fp = f'{data_bihar}/bihar_fw:{FW}_ws:{WS}_test_gnn.pkl'

    locs = load_locs_as_tuples(locs_file)
    train_data = SpatioTemporalDataset(train_fp, locs, HIDDEN_DIM) 
    val_data = SpatioTemporalDataset(val_fp, locs, HIDDEN_DIM)
    test_data = SpatioTemporalDataset(test_fp, locs, HIDDEN_DIM)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    # print(test_data[0])

    input_dim, hidden_dim, output_dim = train_data.get_input_dimensions()
    # print(input_dim, hidden_dim, output_dim)

    model = GNNLSTM(input_dim, hidden_dim, output_dim, device)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train_losses, val_losses = [], []
    print(f"---------\t Training started lr={LR},  hidden_size={hidden_dim} \t---------")
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        model.train()
        for data in train_loader:
            x, y, edge_index = data.x.to(device), data.y[:, -1, :].to(device), data.edge_index.to(device)

            preds = model(x, edge_index)
            
            train_loss = torch.sqrt(criterion(y, preds))

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_losses.append(train_loss.item())

        model.eval()
        with torch.no_grad():
            for data in val_loader:
                x, y, edge_index = data.x.to(device), data.y[:, -1, :].to(device), data.edge_index.to(device)
                preds = model(x, edge_index)
                
                val_loss = torch.sqrt(criterion(y, preds))
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
            x, y, edge_index = data.x.to(device), data.y[:, -1, :].to(device), data.edge_index.to(device)

            preds = model(x, edge_index)

            y_train.extend(y.cpu().tolist())
            preds_train.extend(preds.cpu().tolist())

        with torch.no_grad():
            for data in val_loader:
                x, y, edge_index = data.x.to(device), data.y[:, -1, :].to(device), data.edge_index.to(device)
                preds = model(x, edge_index)

                y_val.extend(y.cpu().tolist())
                preds_val.extend(preds.cpu().tolist())

        for data in test_loader:
            x, y, edge_index = data.x.to(device), data.y[:, -1, :].to(device), data.edge_index.to(device)
            preds = model(x, edge_index)
            
            test_loss = torch.sqrt(criterion(y, preds))

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