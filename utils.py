import numpy as np
import pandas as pd
from scipy import stats
import torch

def eval_stat(y_pred, y, haze_thresh):

    spearmanr = stats.spearmanr(y_pred.ravel(), y.ravel())[0]

    predict_haze = y_pred >= haze_thresh
    predict_clear = y_pred < haze_thresh
    label_haze = y >= haze_thresh
    label_clear = y < haze_thresh

    hit = np.sum(np.logical_and(predict_haze, label_haze))
    miss = np.sum(np.logical_and(label_haze, predict_clear))
    falsealarm = np.sum(np.logical_and(predict_haze, label_clear))

    csi = hit / (hit + falsealarm + miss)
    pod = hit / (hit + miss)
    far = falsealarm / (hit + falsealarm)

    predict = y_pred[:,:,:,0].transpose((0,2,1))
    label = y[:,:,:,0].transpose((0,2,1))
    predict = predict.reshape((-1, predict.shape[-1]))
    label = label.reshape((-1, label.shape[-1]))
    mae = np.mean(np.mean(np.abs(predict - label), axis=1))
    rmse = np.mean(np.sqrt(np.mean(np.square(predict - label), axis=1)))
    return {
        'RMSE': round(rmse, 4), 
        'MAE': round(mae, 4),
        'SpearmanR': round(spearmanr, 4),
        'CSI': round(csi, 4), 
        'POD': round(pod, 4), 
        'FAR': round(far, 4)
    }

def final_stats(stats):
    df = pd.DataFrame(data=stats)
    stats = {col: f'{round(x, 4)} \u00B1 {round(y, 4)}' for (col, x, y) in zip(df.columns, df.mean(axis=0), df.std(axis=0))}
    return stats

def save_model(model, optimizer, fp):
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }

    torch.save(state, fp)

def load_model(fp):
    state = torch.load(fp)
    return state['model_state_dict'], state['optimizer_state_dict']

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.min_val_loss = float('inf')

    def early_stop(self, val_loss):
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_val_loss + self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False