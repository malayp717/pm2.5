import sys
sys.path.append('../')
import os
import yaml
import time
import geopandas as gpd
import pandas as pd
import numpy as np
from plots.map_utils import *
from data.eda_utils import *
from scipy.interpolate import griddata
from pathlib import Path

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(f'{proj_dir}/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# ------------- Config parameters start ------------- #
data_dir = config['dirpath']['data_dir']
plot_dir = config['dirpath']['plots_dir']
pkl_fp = data_dir + config['filepath']['pkl_fp']
map_fp = data_dir + config['filepath']['map_fp']
mask_fp = data_dir + '/mask.txt'

GRID_SIZE = 1_000
AREA_BIHAR = 94_163
DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
# ------------- Config parameters end ------------- #

def monthly_csv(data_ts_dict, mask):

    grid_long, grid_lat = np.meshgrid(np.linspace(min_long, max_long, GRID_SIZE), np.linspace(min_lat, max_lat, GRID_SIZE))

    '''
        monthly_vals is a dictionary with mapping: month -> [pm25, number of observations]
    '''
    monthly_vals = {mnth: [np.zeros((GRID_SIZE, GRID_SIZE)), 0] for mnth in MONTHS}

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
        df.to_csv(f'{data_dir}/{mnth}.csv', index=False)

    print(f'Time taken: {(time.time()-start)/60:.3f} mins')

def hourly_csv(data_ts_dict, mask):

    grid_long, grid_lat = np.meshgrid(np.linspace(min_long, max_long, GRID_SIZE), np.linspace(min_lat, max_lat, GRID_SIZE))

    '''
        hourly_vals is a dictionary with mapping: hour -> [pm25, number of observations]
    '''
    hourly_vals = {hr: [np.zeros((GRID_SIZE, GRID_SIZE)), 0] for hr in range(24)}

    start = time.time()

    for date, row in data_ts_dict.items():
        hr = date.hour
        # print(row)
        grid_values = griddata((row['latitude'], row['longitude']), row['pm25'], (grid_lat, grid_long), method='nearest')
        lcn_val = LCN(grid_long, grid_lat, grid_values)
        hourly_vals[hr][0] = np.add(hourly_vals[hr][0], lcn_val)
        hourly_vals[hr][1] += 1

    for hr, row in hourly_vals.items():
        if row[1] == 0: continue
        pm25 = row[0] / row[1]
        df = pd.DataFrame({'latitude': grid_lat[mask], 'longitude': grid_long[mask], 'pm25': pm25[mask]})
        df.to_csv(f'{data_dir}/{hr}.csv', index=False)

    print(f'Time taken: {(time.time()-start)/60:.3f} mins')

def weekday_weekend_csv(data_ts_dict, mask):

    grid_long, grid_lat = np.meshgrid(np.linspace(min_long, max_long, GRID_SIZE), np.linspace(min_lat, max_lat, GRID_SIZE))

    '''
        week_vals is a dictionary with mapping: weekday / weekend -> [pm25, number of observations]
    '''
    week_vals = {day: [np.zeros((GRID_SIZE, GRID_SIZE)), 0] for day in DAYS}
    weekday_cnt, weekend_cnt = 0, 0
    weekday_pm25, weekend_pm25 = np.zeros((GRID_SIZE, GRID_SIZE)), np.zeros((GRID_SIZE, GRID_SIZE))

    start = time.time()

    for date, row in data_ts_dict.items():
        day = date.day_name()
        grid_values = griddata((row['latitude'], row['longitude']), row['pm25'], (grid_lat, grid_long), method='nearest')
        lcn_val = LCN(grid_long, grid_lat, grid_values)
        week_vals[day][0] = np.add(week_vals[day][0], lcn_val)
        week_vals[day][1] += 1

    for day, row in week_vals.items():
        if row[1] == 0: continue
        
        if day in ['Saturday', 'Sunday']:
            weekend_pm25 += row[0]
            weekend_cnt += row[1]
        else:
            weekday_pm25 += row[0]
            weekday_cnt += row[1]
            
    weekday_pm25, weekend_pm25 = weekday_pm25/weekday_cnt, weekend_pm25/weekend_cnt
            
    weekday_df = pd.DataFrame({'latitude': grid_lat[mask], 'longitude': grid_long[mask], 'pm25': weekday_pm25[mask]})
    weekend_df = pd.DataFrame({'latitude': grid_lat[mask], 'longitude': grid_long[mask], 'pm25': weekend_pm25[mask]})

    weekday_df.to_csv(f'{data_dir}/weekday.csv', index=False)
    weekend_df.to_csv(f'{data_dir}/weekend.csv', index=False)

    print(f'Time taken: {(time.time()-start)/60:.3f} mins')

def seasonal_csv():
    seasons = {
        'JJAS': ['June', 'July', 'August', 'September'],
        'ON': ['October', 'November'],
        'DJF': ['December', 'January', 'February'],
        'MA': ['March', 'April'] 
    }

    start = time.time()

    for name, season in seasons.items():
        df = None

        for mnth in season:
            mnth_fp = f'{data_dir}/{mnth}.csv'
            df_t = pd.read_csv(mnth_fp)
            df_t.sort_values(by=['latitude', 'longitude'])

            if df is None:
                df = df_t.copy(deep=True)
            else:
                df['pm25'] += df_t['pm25']
        
        df['pm25']/=len(season)
        df.to_csv(f'{data_dir}/{name}.csv', index=False)

    print(f'Time taken: {(time.time()-start)/60:.3f} mins')

def plots(bihar, file):

    start_time = time.time()

    df = pd.read_csv(f'{data_dir}/{file}.csv')
    # print(df.head())

    grid_long, grid_lat, pm25 = df['longitude'], df['latitude'], df['pm25']
    # print(len(grid_long), len(grid_lat), len(pm25))

    create_plot(grid_long, grid_lat, pm25, bihar, f'{plot_dir}/map_LCN_{file}_absolute', 'absolute')

    print(f'{file}: {(time.time()-start_time)/60:.3f} mins')

if __name__ == '__main__':

    bihar = gpd.read_file(map_fp)
    df = pd.read_pickle(pkl_fp)
    df['pm25'] = df['pm25'].astype(np.float64)
    df = df[['timestamp', 'longitude', 'latitude', 'pm25']]
    df['timestamp'] = df['timestamp'].astype('datetime64[ns]')

    min_lat, max_lat, min_long, max_long = coordinate_bounds(bihar)
    # print(min_lat, max_lat, min_long, max_long)

    '''
        data_ts_dict: Dictionary that maps timestamp values to corresponding readings
        Example: Timestamp('2023-05-01 00:00:00'): [timestamp, longitude, latitude, pm25]
    '''
    data_ts_dict = {timestamp: group for timestamp, group in df.groupby('timestamp')}

    '''
        Uncomment the following lines only if you want to change the resolution of the map
        Change the value of GRID_SIZE in constants.py to change the resolution of map as well
    '''
    grid_long, grid_lat = np.meshgrid(np.linspace(min_long, max_long, GRID_SIZE), np.linspace(min_lat, max_lat, GRID_SIZE))
    if Path(mask_fp).is_file():
        mask = np.loadtxt(mask_fp, dtype=bool, delimiter='\t')
    else:
        mask = get_indices(grid_long, grid_lat, bihar, mask_fp)

    # weekday_weekend_csv(data_ts_dict, mask)
    # monthly_csv(data_ts_dict, mask)
    # hourly_csv(data_ts_dict, mask)
    # # NOTE: Ensure monthly csv's are already generated before running seasonal csv
    # seasonal_csv()

    files = ['JJAS', 'ON', 'DJF', 'MA']

    for f in files:
        plots(bihar, f)

    print(f'Resolution of maps: {np.sqrt(AREA_BIHAR/np.sum(mask))}')