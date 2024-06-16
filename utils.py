import math
import numpy as np
import pandas as pd
from scipy import stats
from math import radians, sin, cos, sqrt, asin
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch

def haversine_dist(loc1, loc2):
    R = 6371

    lon1, lat1, lon2, lat2 = loc1[0], loc1[1], loc2[0], loc2[1]
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dLon = lon2 - lon1
    dLat = lat2 - lat1
    a = sin(dLat/2) ** 2 + cos(lat1) * cos(lat2) * sin(dLon/2) ** 2
    c = 2 * asin(sqrt(a))

    return R * c

def eval_stat(y_pred, y, haze_thresh):

    RMSE = round(math.sqrt(mean_squared_error(y_pred, y)), 4)
    MAE = round(mean_absolute_error(y, y_pred), 4)
    spearmanr = round(stats.spearmanr(y_pred, y)[0], 4)
    p_value = round(stats.spearmanr(y_pred, y)[1], 4)
    pearsonr = round(stats.pearsonr(y_pred, y)[0], 4)

    pred_haze = y_pred >= haze_thresh
    pred_clear = y_pred < haze_thresh
    label_haze = y >= haze_thresh
    label_clear = y < haze_thresh

    hit = np.sum(np.logical_and(pred_haze, label_haze))
    miss = np.sum(np.logical_and(label_haze, pred_clear))
    false_alarm = np.sum(np.logical_and(pred_haze, label_clear))

    csi = hit / (hit + false_alarm + miss)
    pod = hit / (hit + miss)
    far = false_alarm / (hit + false_alarm)

    csi, pod, far = round(csi, 4), round(pod, 4), round(far, 4)

    return {
        'RMSE': RMSE,
        'MAE': MAE,
        'Spearman R': spearmanr,
        'p value': p_value,
        'Pearson R': pearsonr,
        'CSI': csi,
        'POD': pod,
        'FAR': far
    }

def save_model(model, optimizer, train_loss, val_loss, fp):
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }

    torch.save(state, fp)

def load_model(fp):
    state = torch.load(fp)
    return state['model_state_dict'], state['optimizer_state_dict'], state['train_loss'], state['val_loss']