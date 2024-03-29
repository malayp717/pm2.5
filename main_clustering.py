import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import Point, Polygon
import geopandas as gpd
from scipy.interpolate import interp1d
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')
from constants import *
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        loc, data = row['loc'], row['pm25']
        X, y = np.array(data[:-1]).reshape(-1,1), np.array(data[1:]).reshape(-1,1)
        # X, y = data[:-1], data[-1].reshape(1,)
        # return torch.from_numpy(X), torch.from_numpy(y)
        return torch.Tensor(loc), torch.Tensor(X), torch.Tensor(y)
    
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, out_dim, bidirectional):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        # self.gelu = nn.GELU()
        self.fc = nn.Linear(hidden_dim*2, out_dim)

    def forward(self, x):
        out, (h, _) = self.lstm(x)
        # out = self.gelu(out)
        # out = out[:, -1, :]

        out = self.fc(out)
        return out, h
    
def train(season, train_loader, test_loader, input_size, output_size, hidden_size, num_layers, NUM_EPOCHS, LR):

    model = LSTM(input_size, hidden_size, num_layers, output_size, bidirectional=True)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    model_path = f'{model_dir}/BLSTM_clustering_{season}_{hidden_size}.pth.tar'
    model_file = Path(model_path)

    start_time = time.time()
    start_epoch, train_losses, val_losses = 0, [], []

    if model_file.is_file():
        state = torch.load(model_path)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch, train_losses, val_losses = state['epoch'], state['train_losses'], state['val_losses']

    if start_epoch < NUM_EPOCHS:

        print(f"---------\t Training started lr={LR},  hidden_size={hidden_size}, num_layers={num_layers} \t---------")
        for epoch in range(start_epoch, NUM_EPOCHS):

            loss_train, loss_val = [], []

            for _, inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                preds, _ = model(inputs)

                labels, preds = torch.squeeze(labels), torch.squeeze(preds)
                train_loss = torch.sqrt(criterion(labels, preds))

                train_loss.backward()
                optimizer.step()

                loss_train.append(train_loss.item())
        
            with torch.no_grad():
                for _, inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    preds, _ = model(inputs)
                    labels, preds = torch.squeeze(labels), torch.squeeze(preds)
                    val_loss = torch.sqrt(criterion(labels, preds))

                    loss_val.append(val_loss.item())

            train_loss, val_loss = np.mean(loss_train), np.mean(loss_val)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses
            }

            torch.save(state, model_path)

            if (epoch+1)%50 == 0:
                print(f'Epoch: {epoch+1}/{NUM_EPOCHS}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, \
                    time_taken: {(time.time()-start_time)/60:.2f} mins')
            
        print(f"---------\t Training completed lr={LR},  hidden_size={hidden_size}, num_layers={num_layers} \t---------")

    X, locs = [], []
    
    with torch.no_grad():
        for loc, inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            bs = inputs.shape[0]

            _, h = model(inputs)
            h = torch.permute(h, (1,0,2))
            h = h.cpu().detach().numpy().reshape(bs, -1)
            X.extend(h)
            locs.extend(loc.cpu().detach().numpy())

        for loc, inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            bs = inputs.shape[0]

            _, h = model(inputs)
            h = torch.permute(h, (1,0,2))
            h = h.cpu().detach().numpy().reshape(bs, -1)
            X.extend(h)
            locs.extend(loc.cpu().detach().numpy())

    return np.array(X), np.array(locs)

def seasonal_training(df, locs, season):

    print(f"---------\t Season Info: {season[0]},  start_date={season[1]}, end_date={season[2]} \t---------")

    df = df[(df['timestamp'] >= season[1]) & (df['timestamp'] < season[2])]
    print(f'Data Shape: {df.shape}')
    df_grouped = df.groupby(['latitude', 'longitude'])

    data = []

    for loc, group in df_grouped:
        data.append({'loc': loc, 'pm25': group['pm25'].tolist()})

    locs = list(locs)
    train_locs, test_locs = locs[:len(locs) // 2], locs[len(locs) // 2:]
    print(len(locs), len(train_locs), len(test_locs))

    train_data, test_data = [], []
    
    for row in data:
        if row['loc'] in train_locs: train_data.append(row)
        else: test_data.append(row)
    
    BATCH_SIZE, LR, NUM_EPOCHS = 64, 1e-3, [700, 600, 500]
    INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE = None, [32, 64, 128], 2, None

    train_dataset = CustomDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_dataset = CustomDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
                
    for num_epochs, hs in zip(NUM_EPOCHS, HIDDEN_SIZE):
        for _, inputs, labels in train_loader:
            INPUT_SIZE, OUTPUT_SIZE = inputs.shape[-1], labels.shape[-1]
            INPUT_SHAPE, OUTPUT_SHAPE = inputs.shape, labels.shape
            break

        print(INPUT_SHAPE, OUTPUT_SHAPE, INPUT_SIZE, OUTPUT_SIZE)

        embeddings, locs = train(season[0], train_loader, test_loader, INPUT_SIZE, OUTPUT_SIZE, hs, NUM_LAYERS, num_epochs, LR)

        data = {}
        for loc, emb in zip(locs, embeddings):
            data[(loc[0], loc[1])] = emb

        with open(f'{data_bihar}/bihar_embedding_{season[0]}_{hs}.pkl', 'wb') as f:
            pickle.dump(data, f)

if __name__ == '__main__':

    bihar = gpd.read_file(f'{data_bihar}/bihar.json')
    file = f'{data_bihar}/AMRIT_PM25_DATA.csv'
    df = pd.read_csv(file)
    df = df[df['State'] == 'BIHAR']
    df = df[[x for x in df.columns if x not in {'Deviceid', 'State', 'Block', 'District', 'Devicemfg', 'Remark'}]]

    cols = {'timestamp': 'datetime64[ns]',  'latitude': np.float64, 'longitude': np.float64, 'pm25': np.float64}
    ts = [pd.Timestamp(x) for x in df.columns if x not in {'latitude', 'longitude'}]

    df_new = pd.DataFrame(columns=cols)
    locs = set()

    for _, row in df.iterrows():
        loc, pm25 = row.to_numpy()[:2], row.to_numpy()[2:]
        lat, lon = loc[0], loc[1]

        if np.count_nonzero(np.isnan(pm25)) >= 6_000: continue

        if (lat, lon) not in locs: locs.add((lat, lon))
        else: continue

        lat, lon = [loc[0]] * len(pm25), [loc[1]] * len(pm25)

        df_temp = pd.DataFrame(columns=cols)
        df_temp['timestamp'] = ts
        df_temp['latitude'] = lat
        df_temp['longitude'] = lon
        df_temp['pm25'] = pm25
        
        df_new = pd.concat([df_new, df_temp])
    
    df_temp = df_new.copy(deep=True)
    df_temp['timestamp'] = df_temp['timestamp'].values.astype(float)

    data_new = df_temp.to_numpy()
    imputer = IterativeImputer(random_state=0)
    data_new = imputer.fit_transform(data_new)

    df_new['pm25'] = data_new[:, -1].clip(0, 500)
    # print(df_new.head())

    seasons = [['monsoon', pd.Timestamp(year=2023, month=6, day=1), pd.Timestamp(year=2023, month=10, day=1)],\
               ['post_monsoon', pd.Timestamp(year=2023, month=10, day=1), pd.Timestamp(year=2023, month=12, day=1)],\
                ['winter', pd.Timestamp(year=2023, month=12, day=1), pd.Timestamp(year=2024, month=3, day=1)],\
                ['combined', pd.Timestamp(year=2023, month=10, day=1), pd.Timestamp(year=2024, month=3, day=1)]]

    for season in seasons:
        seasonal_training(df_new, locs, season)

    # # print(embeddings.shape, locs.shape)

    # # lats = [x[0] for x in locs]
    # # lons = [x[1] for x in locs]

    # # scores = []
    # # K = [i for i in range(2, 21)]

    # # for k in K:
    # #     kmeans = KMeans(n_clusters=k).fit(embeddings)
    # #     scores.append(kmeans.inertia_)

    # # plt.plot(K, scores)
    # # plt.xticks(K, rotation='vertical')
    # # plt.savefig(f'{plot_dir}/elbow.jpg', dpi=400)

    # # labels = KMeans(n_clusters=8).fit_predict(X)

    # # _, ax = plt.subplots(figsize=(10,8))
    # # # colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:gray', 'tab:olive']
    # # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    # # bihar.plot(ax=ax, color='white', edgecolor='grey', linewidth=0.5)
    # # scatter = ax.scatter(lons, lats, c=labels, cmap=matplotlib.colors.ListedColormap(colors), marker='o')

    # # ax.set_axis_off()
    # # plt.savefig(f'{plot_dir}/clusters.jpg', dpi=400)
    # # plt.close()