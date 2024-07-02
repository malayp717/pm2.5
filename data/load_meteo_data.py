import sys
import os
sys.path.append('..')
import numpy as np
import pandas as pd
import yaml
from datetime import timedelta
from itertools import product
from data.MeteoDataset import MeteoDataset
import time

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(f'{proj_dir}/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# ------------- Config parameters start ------------- #
location = 'bihar'
data_dir = config['dirpath']['data_dir']
locs_fp = data_dir + config[location]['filepath']['locations_fp']
data_dir = config['dirpath']['data_dir']

METEO_COLUMNS_DICT = {'timestamp': 'datetime64[ns]',
                      'longitude': float,
                      'latitude': float,
                      'rh': float,
                      'temp': float,
                      'pm25': float}
# ------------- Config parameters end ------------- #

def superset(start_date, end_date, locs):
    ts = []
    while start_date < end_date:
        ts.append(start_date)
        start_date += timedelta(hours=1)

    df = pd.DataFrame(list(product(ts, locs)), columns=['timestamp', 'locs'])
    df['longitude'] = df['locs'].apply(lambda x: x[1])
    df['latitude'] = df['locs'].apply(lambda x: x[0])
    df = df[['timestamp', 'longitude', 'latitude']]

    return df

def data_proc(df, ts, start_index):
    
    locs = {}
    p_df = pd.DataFrame(columns=[x for x in METEO_COLUMNS_DICT.keys()])
    p_df = p_df.astype(METEO_COLUMNS_DICT)

    for i, col in enumerate(df.columns):

        if i == 0: continue
        option = 1 if col[0] == 'P' else (2 if col[0] == 'T' else 3)
        values = np.array(df[col][start_index:], dtype=np.float64)

        df[col][2], df[col][3] = float(df[col][2]), float(df[col][3])
        loc = (df[col][2], df[col][3])

        if loc not in locs:
            locs[loc] = MeteoDataset(ts, df[col][2], df[col][3])
            locs[loc].set_features(values, option)
        else:
            locs[loc].set_features(values, option)
            
    for _, obj in locs.items():

        dtype = {'timestamp': 'datetime64[ns]', 'rh': float, 'temp': float, 'pm25': float}
        t_df = pd.DataFrame(columns=[x for x in dtype.keys()])
        t_df = t_df.astype(dtype)

        t_df['timestamp'] = obj.timestamp
        t_df['rh'] = obj.rh
        t_df['temp'] = obj.temp
        t_df['pm25'] = obj.pm25

        '''
            15 min updates -> Hourly Updates.
        '''
        t_df['timestamp'] = t_df['timestamp'].dt.floor('H')
        numeric_cols = [col for col in t_df.columns if t_df[col].dtype == float]

        t_df = t_df.groupby('timestamp')[numeric_cols].mean()
        t_df = t_df.reset_index()
        '''
            15 min updates -> Hourly Updates completed
        '''
        
        t_df['latitude'] = [obj.latitude] * t_df.shape[0]
        t_df['longitude'] = [obj.longitude] * t_df.shape[0]

        p_df = pd.concat([p_df, t_df], ignore_index=True)
    
    return p_df

def create_meteo_dataframe(files, file_name, start_date, end_date):

    locs_df = pd.read_csv(locs_fp, sep='|', header=None)
    bihar_locs = set(locs_df.apply(lambda x: (float(x[4]), float(x[3])), axis=1))

    df = pd.DataFrame(columns=[x for x in METEO_COLUMNS_DICT.keys()])
    df = df.astype(METEO_COLUMNS_DICT)

    for f, start_index in files.items():

        start_time = time.time()

        f_df = pd.read_csv(f'{data_dir}/{f}')
        ts = pd.to_datetime(f_df.to_numpy()[start_index:, 0])

        t_df = data_proc(f_df, ts, start_index)
        df = pd.concat([df, t_df], ignore_index=True)

        print(f'{f} processed \t time_taken: {(time.time()-start_time)/60:.2f} mins')

    df = df.sort_values(by='timestamp')

    df = df[df['timestamp'] < end_date]
    duration = (end_date - start_date).days * 24

    df_g = df.groupby(['latitude', 'longitude'])
    locs = []

    for loc, group in df_g:
        if loc not in bihar_locs: continue
        if group.shape[0] != duration: continue
        locs.append(loc)

    df = df[df[['latitude', 'longitude']].apply(tuple, 1).isin(locs)]
    df_loc = superset(start_date, end_date, list(bihar_locs))

    df = pd.merge(df, df_loc, on=['timestamp', 'longitude', 'latitude'], how='outer')
    df = df.sort_values(by=['timestamp', 'longitude', 'latitude'])
    df.to_pickle(file_name, protocol=4)
    
    return df