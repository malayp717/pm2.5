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