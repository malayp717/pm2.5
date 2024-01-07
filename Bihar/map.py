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

def monthly_csv(data_ts_dict, mask):

    grid_long, grid_lat = np.meshgrid(np.linspace(min_long, max_long, GRID_SIZE), np.linspace(min_lat, max_lat, GRID_SIZE))

    '''
        monthly_vals is a dictionary with mapping: month -> [pm25, number of observations]
    '''
    monthly_vals = {mnth: [np.zeros((GRID_SIZE, GRID_SIZE)), 0] for mnth in months}

    start = time.time()

    for date, row in data_ts_dict.items():
        mnth = date.strftime('%B')
        # print(row)
        grid_values = griddata((row['latitude'], row['longitude']), row['pm25'], (grid_lat, grid_long), method='nearest')
        lcn_val = LCN(grid_long, grid_lat, grid_values)
        monthly_vals[mnth][0] = np.add(monthly_vals[mnth][0], lcn_val)
        monthly_vals[mnth][1] += 1

    for mnth, row in monthly_vals.items():
        if row[1] == 0: continue
        pm25 = row[0] / row[1]
        df = pd.DataFrame({'latitude': grid_lat[mask], 'longitude': grid_long[mask], 'pm25': pm25[mask]})
        df.to_csv(f'{data_bihar}/{mnth}.csv', index=False)

    print(f'Time taken: {time.time()-start:.3f} s')


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
    df = df[['timestamp', 'longitude', 'latitude', 'rh', 'temp', 'pm25']]
    # print(df.dtypes)
    # print(df.head())

    min_lat, max_lat, min_long, max_long = coordinate_bounds(bihar)
    # print(min_lat, max_lat, min_long, max_long)

    '''
        data_ts_dict: Dictionary that maps timestamp values to corresponding readings
        Example: Timestamp('2023-05-01 00:00:00'): [timestamp, longitude, latitude, rh, temp, pm25]
    '''
    data_ts_dict = {timestamp: group for timestamp, group in df.groupby('timestamp')}

    '''
        Uncomment the following lines only if you want to change the resolution of the map
        Change the value of GRID_SIZE in constants.py to change the resolution of map as well
    '''
    # mask = get_indices(grid_long, grid_lat, bihar)
    # np.savetxt(f'{data_bihar}/mask.txt', mask, fmt='%d', delimiter='\t')
    mask = np.loadtxt(f'{data_bihar}/mask.txt', dtype=bool, delimiter='\t')

    monthly_csv(data_ts_dict, mask)
    monthly_plots(bihar)
    print(f'Resolution of maps: {np.sqrt(AREA_BIHAR/np.sum(mask))}')