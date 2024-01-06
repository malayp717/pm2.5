import sys
sys.path.append('../')
import argparse
import time
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from constants import *
from map_utils import *
from eda_utils import *
from scipy.interpolate import griddata

def monthly_csv(data_dt_dict, grid_long, grid_lat, mask, bihar):

    monthly_eval = {}

    for date, ts_dict in data_dt_dict.items():
        # dt = date.strftime('%Y-%m-%d')
        mnth = date.strftime('%B')
        tot_values = np.zeros((GRID_SIZE, GRID_SIZE))
        
        for data in ts_dict:
            for _, val in data.items():
                grid_values = griddata((val['latitude'], val['longitude']), val['pm25'], (grid_lat, grid_long), method='nearest')
                lcn_val = LCN(grid_long, grid_lat, grid_values)
                # mean, var = np.mean(lcn_val), np.var(lcn_val)
                # lcn_val = (lcn_val - mean) / np.sqrt(var)
                tot_values = np.add(tot_values, lcn_val)
        
        tot_values = tot_values / len(data)
        
        if mnth not in monthly_eval:
            monthly_eval[mnth] = [tot_values]
        else:
            monthly_eval[mnth].append(tot_values)

        if mnth != 'May':
            break
    
    for mnth in monthly_eval.keys():
        print(len(monthly_eval[mnth]))
        pm25 = np.mean(np.array(monthly_eval[mnth]), axis=0)

        # Write to CSV file
        df = pd.DataFrame({'latitude': grid_lat[mask], 'longitude': grid_long[mask], 'pm25': tot_values[mask]})
        df.to_csv(f'{data_bihar}/{mnth}.csv', index=False)
        break


def monthly_plots(bihar):
    mnths = ['May', 'June', 'July', 'August', 'September', 'October', 'November']

    start_time = time.time()

    for mnth in mnths:
        df = pd.read_csv(f'{data_bihar}/{mnth}.csv')
        # print(df.head())

        grid_long, grid_lat, pm25 = df['longitude'], df['latitude'], df['pm25']
        # print(len(grid_long), len(grid_lat), len(pm25))

        create_plot(grid_long, grid_lat, pm25, bihar, f'{bihar_plot_dir}/map_LCN_{mnth}_absolute', 'absolute')
        create_plot(grid_long, grid_lat, pm25, bihar, f'{bihar_plot_dir}/map_LCN_{mnth}_relative', 'relative')

        print(f'{mnth}: {time.time()-start_time} s')
        start_time = time.time()

if __name__ == '__main__':

    bihar = gpd.read_file(f'{data_bihar}/bihar.json')
    data_file = f'{data_bihar}/bihar_512_sensor_data_imputed.pkl'

    df = pd.read_pickle(data_file)
    df['pm25'] = df['pm25'].astype(np.float64)
    # print(df.dtypes)

    min_lat, max_lat, min_long, max_long = coordinate_bounds(bihar)
    # print(min_lat, max_lat, min_long, max_long)

    data_ts_dict = {timestamp: group for timestamp, group in df.groupby('timestamp')}
    data_dt_dict = {}

    for timestamp, values in data_ts_dict.items():
        date = timestamp.date()
        row = {timestamp: values}
        if date not in data_dt_dict:
            data_dt_dict[date] = []
        data_dt_dict[date].append(row)
    
    # print(data_dt_dict.keys())

    pm25_values = []
    grid_long, grid_lat = np.meshgrid(np.linspace(min_long, max_long, GRID_SIZE), np.linspace(min_lat, max_lat, GRID_SIZE))

    '''
        Uncomment the following lines only if you want to change the resolution of the map
        Change the value of GRID_SIZE in constants.py to change the resolution of map as well
    '''
    # mask = get_indices(grid_long, grid_lat, bihar)
    # np.savetxt(f'{data_bihar}/mask.txt', mask, fmt='%d', delimiter='\t')
    mask = np.loadtxt(f'{data_bihar}/mask.txt', dtype=bool, delimiter='\t')

    monthly_csv(data_dt_dict, grid_long, grid_lat, mask ,bihar)
    # monthly_plots(bihar)
    print(f'Resolution of maps: {np.sqrt(AREA_BIHAR/np.sum(mask))}')