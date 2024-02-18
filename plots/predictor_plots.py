import sys
sys.path.append('../')
import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import metrics
import pandas as pd
from constants import *
from utils import *
from sklearn.preprocessing import StandardScaler
# from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')

def eval_stat(y_train_pred, y_train):
    Rsquared = stats.spearmanr(y_train_pred, y_train.ravel())[0]
    pvalue = stats.spearmanr(y_train_pred, y_train.ravel())[1]
    Rsquared_pearson = stats.pearsonr(y_train_pred, y_train.ravel())[0]
    pvalue_pearson = stats.pearsonr(y_train_pred, y_train.ravel())[1]
    return Rsquared, pvalue, Rsquared_pearson, pvalue_pearson

def calculateSpatial(y_test_pred, y_test, test_stations):
    df = pd.DataFrame({'y_test': y_test, 'y_test_pred': y_test_pred, 'test_stations': test_stations}).groupby(['test_stations']).mean()
    test_station_avg_pred = np.array(df.y_test_pred)
    test_station_avg = np.array(df.y_test)
    _, _, Rsquared_pearson, _ = eval_stat(test_station_avg_pred, test_station_avg)
    rmse = np.sqrt(metrics.mean_squared_error(test_station_avg, test_station_avg_pred))
    return Rsquared_pearson, rmse, test_station_avg_pred, test_station_avg

def plot_result(y_pred, y_true, Rsquared, pvalue, Rsquared_pearson, pvalue_pearson, plot_label="train", save=True, 
                fig_name="", lower_bound=0, upper_bound=100, spatial_R=-1, station_name=None):
    plt.clf()
    mean = np.mean(y_true)
    fig, ax = plt.subplots(figsize=(12, 10))
    data = pd.DataFrame(data={'y_true': y_true, 'y_pred': y_pred})
    ax = sns.histplot(data, x='y_true', y='y_pred', cbar=True, color='orange', kde=True, element='step')
    ax.plot([lower_bound, upper_bound], [lower_bound, upper_bound], color='grey', linestyle = 'dashed', marker= '.', lw=2)
    ax.set_xlabel('True $PM_{2.5}$ ($\mu $g m$^{-3}$)', size=20)
    ax.set_ylabel('Predicted $PM_{2.5}$ ($\mu $g m$^{-3}$)', size=20)
    ax.tick_params(labelsize=15)
    #ax.legend(prop={'size': 20})
    ax.text(0.02, 0.98, 'Spearman r = '+ str(round(Rsquared,2)), ha='left', va='top', color='black', weight='roman', fontsize=16, transform=ax.transAxes)
    ax.text(0.02, 0.94, 'Spearman p-value = '+ str(round(pvalue,2)), ha='left', va='top', color='black', weight='roman', fontsize=16, transform=ax.transAxes)
    ax.text(0.02, 0.90, 'Pearson r = '+ str(round(Rsquared_pearson,2)), ha='left', va='top', color='black', weight='roman', fontsize=16, transform=ax.transAxes)
    ax.text(0.02, 0.86, 'Pearson p-value = '+ str(round(pvalue_pearson,3)), ha='left', va='top', color='black', weight='roman', fontsize=16, transform=ax.transAxes)
    ax.text(0.02, 0.82, 'RMSE = '+ str(round(np.sqrt(metrics.mean_squared_error(y_true, y_pred)),2)), ha='left', va='top', color='black', weight='roman', fontsize=16, transform=ax.transAxes)
    ax.text(0.02, 0.78, 'NRMSE = '+ str(round(np.sqrt(metrics.mean_squared_error(y_true, y_pred))/mean,2)), ha='left', va='top', color='black', weight='roman', fontsize=16, transform=ax.transAxes)
    ax.text(0.02, 0.74, 'MAE = '+ str(round(metrics.mean_absolute_error(y_true, y_pred),2)), ha='left', va='top', color='black', weight='roman', fontsize=16, transform=ax.transAxes)
    ax.text(0.02, 0.70, 'NMAE = '+ str(round(metrics.mean_squared_error(y_true, y_pred)/mean,2)), ha='left', va='top', color='black', weight='roman', fontsize=16, transform=ax.transAxes)
    ax.text(0.02, 0.66, '% error = '+ str(round(metrics.mean_absolute_error(y_true, y_pred)/np.mean(y_true)*100,1))+'%', ha='left', va='top', color='black', weight='roman', fontsize=16, transform=ax.transAxes)
    
    if spatial_R != -1:
        ax.text(0.02, 0.62, 'Spatial R = ' + str(round(spatial_R, 2)), ha='left', va='top', color='black', weight='roman', fontsize=16, transform=ax.transAxes)
    if plot_label == "train":
        if station_name is not None:
            ax.text(0.65, 0.10, station_name + ' (train)', bbox=dict(facecolor='grey', alpha=0.9), ha="left", va="top", color='black', weight='roman', fontsize=20, transform=ax.transAxes)
        else:
            ax.text(0.65, 0.10, 'All Stations' + ' (train)', bbox=dict(facecolor='grey', alpha=0.9), ha="left", va="top", color='black', weight='roman', fontsize=20, transform=ax.transAxes)
    else:
        if station_name is not None:
            ax.text(0.65, 0.10, station_name + ' (test)', bbox=dict(facecolor='grey', alpha=0.9), ha="left", va="top", color='black', weight='roman', fontsize=20, transform=ax.transAxes)
        else:
            ax.text(0.65, 0.10, 'All Stations' + ' (test)', bbox=dict(facecolor='grey', alpha=0.9), ha="left", va="top", color='black', weight='roman', fontsize=20, transform=ax.transAxes)        
    # plt.gca().set_aspect('equal', adjustable='box')
    if save:
        plt.savefig(f'{plot_dir}/{fig_name}.jpg', dpi=300)
    plt.show()
    del data, ax
    return

def spatialRPlot(color, y_test_ref,  y_test_ref_pred_raw, plot_label = 'test', 
                 save=False, fig_name="", line_range=[50, 150], station_name=None):
    plt.clf()
    Rsquared, pvalue, Rsquared_pearson, pvalue_pearson = eval_stat(y_test_ref_pred_raw, y_test_ref)
    y_train_pred_mlpr,y_train, Rsquared, pvalue, Rsquared_pearson, pvalue_pearson = y_test_ref_pred_raw, y_test_ref, Rsquared, pvalue, Rsquared_pearson, pvalue_pearson
        
    plt.rcParams.update({'mathtext.default':  'regular' })
    my_prediction = y_train_pred_mlpr
    fig, ax = plt.subplots(figsize = (8,8))
    ax.scatter(y_train, my_prediction, color = color,alpha =1, edgecolors='navy',  s = 100)
    ax.plot(line_range, line_range, 'k--', lw=4)
    ax.set_xlabel('True $PM_{2.5}$ ($\mu $g m$^{-3}$)', size = 25)
    ax.set_ylabel('Predicted $PM_{2.5}$ ($\mu $g m$^{-3}$)', size = 25)
    ax.tick_params(labelsize = 25)
    horozontal_ax = 0.05
    vertical_offset = 0.2
    ax.text(horozontal_ax, 0.72+vertical_offset, 'Pearson r = '+ str(round(Rsquared_pearson,2)), color='black', weight='roman',
    fontsize=25,transform=ax.transAxes)
    ax.text(horozontal_ax, 0.65+vertical_offset, 'p-value = '+ str(round(pvalue_pearson,3)), color='black', weight='roman',
    fontsize=25,transform=ax.transAxes)
    ax.text(horozontal_ax, 0.58+vertical_offset, 'RMSE = '+ str(round(np.sqrt(metrics.mean_squared_error(y_train, my_prediction)),2)), 
    color='black', weight='roman', fontsize=25, transform=ax.transAxes)
   
    if plot_label == "train":
        if station_name is not None:
            ax.text(0.1, 0.4, station_name + ' (train)', bbox=dict(facecolor='grey', alpha=0.9),color='black', weight='roman',
        fontsize=25,transform=ax.transAxes)
        else:
            ax.text(0.4, 0.1, 'All stations' + ' (train)', bbox=dict(facecolor='grey', alpha=0.9),color='black', weight='roman',
        fontsize=25,transform=ax.transAxes)
    else:
        if station_name is not None:
            ax.text(0.4, 0.1, station_name + ' (test)', bbox=dict(facecolor='grey', alpha=0.9),color='black', weight='roman',
        fontsize=25,transform=ax.transAxes)
        else:
            ax.text(0.4, 0.1, 'All stations' + ' (test)', bbox=dict(facecolor='grey', alpha=0.9),color='black', weight='roman',
        fontsize=25,transform=ax.transAxes)
    plt.tight_layout()
    if save:
        plt.savefig(f'{plot_dir}/{fig_name}.jpg', dpi=300)
    pass
    del fig, ax
    return

def spatialR_over_time(df, model):

    timestamps, spatial_R_values = [], []
    cols = df.columns
    df['locs'] = list(zip(df['latitude'], df['longitude']))
    ts = df['timestamp'].to_list()
    df['timestamp'] = df['timestamp'].values.astype(float)

    test_locs = load_locs_as_tuples(f'{data_bihar}/test_locations.txt')
    df = df[df['locs'].isin(test_locs)]

    cols = ['timestamp', 'latitude', 'longitude', 'rh', 'temp', 'blh', 'u10', 'v10', 'kx', 'sp', 'tp', 'pm25']
    df = df[cols]
    data = df[[x for x in cols if x != 'pm25']].to_numpy()

    start_time = time.time()
    print('******\t\t Spatial R over time \t\t******')

    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    df[[x for x in cols if x != 'pm25']] = data

    df_grouped = df.groupby('timestamp')
    ts_info = {y: x for x, y in zip(ts, data[:, 0])}

    for key, group in df_grouped:

        X, y = group.to_numpy()[: , :-1], group.to_numpy()[:, -1]
        test_stations = [(x, y) for x, y in zip(X[:, 1], X[:, 2])]
        y_pred = np.clip(model.predict(X), LOWER_BOUND, UPPER_BOUND)
        spatial_R, _, _, _ = calculateSpatial(y_pred, y, test_stations)
        spatial_R_values.append(spatial_R)
        timestamps.append(ts_info[key])

    plt.plot(timestamps, spatial_R_values)
    plt.savefig(f'{plot_dir}/spatial_R_over_time.jpg', dpi=300)

    print(f'Process Completed\nTime taken: {time.time()-start_time:.2f} s')

def pred_plots(df, model):

    X_train, y_train, _, _, X_test, y_test = train_test_split(df, cols, split_ratio=[0.4,0.1,0.5], split_type='lat_long', normalize=True)
    train_stations, test_stations = [(x, y) for x, y in zip(X_train[:, 1], X_train[:, 2])], [(x, y) for x, y in zip(X_test[:, 1], X_test[:, 2])]

    y_train_pred, y_test_pred = model.predict(X_train), model.predict(X_test)
    y_train_pred, y_test_pred = np.clip(y_train_pred, LOWER_BOUND, UPPER_BOUND), np.clip(y_test_pred, LOWER_BOUND, UPPER_BOUND) 

    spatial_R, spatial_rmse, station_avg_pred, station_avg = calculateSpatial(y_train_pred, y_train, train_stations)
    Rsquared, pvalue, Rsquared_pearson, pvalue_pearson = eval_stat(y_train_pred, y_train)


    start_time = time.time()
    print('******\t\t Plots for training data \t\t******')

    plot_result(y_train_pred, y_train, Rsquared, pvalue, Rsquared_pearson, pvalue_pearson, plot_label='train', save=True, 
                fig_name='PM2.5_XGB_train_stations', lower_bound=0, upper_bound=400, spatial_R=spatial_R)
    spatialRPlot('dodgerblue', station_avg, station_avg_pred, plot_label='train', save=True, 
                fig_name='PM2.5_XGB_train_spatial_R')
    
    print(f'Process Completed\nTime taken: {time.time()-start_time:.2f} s')

    start_time = time.time()
    print('******\t\t Plots for test data \t\t******')

    plot_result(y_test_pred, y_test, Rsquared, pvalue, Rsquared_pearson, pvalue_pearson, plot_label='test', save=True, 
                fig_name='PM2.5_XGB_test_stations', lower_bound=0, upper_bound=400, spatial_R=spatial_R)
    spatialRPlot('dodgerblue', station_avg, station_avg_pred, plot_label='test', save=True, 
                fig_name='PM2.5_XGB_test_spatial_R')
    
    print(f'Process Completed\nTime taken: {time.time()-start_time:.2f} s')

if __name__ == '__main__':

    cols = ['timestamp', 'latitude', 'longitude', 'rh', 'temp', 'blh', 'u10', 'v10', 'kx', 'sp', 'tp', 'pm25']
    data_file = f'{data_bihar}/bihar_512_sensor_era5_image_imputed.pkl'
    model_file = f'{model_dir}/bihar_xgb_iterative_lat_long.pkl'

    df = pd.read_pickle(data_file)
    df['pm25'] = df['pm25'].clip(LOWER_BOUND, UPPER_BOUND)
    model = joblib.load(model_file)

    # pred_plots(df, model)
    spatialR_over_time(df, model)