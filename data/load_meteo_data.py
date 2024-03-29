import sys
sys.path.append('..')
import numpy as np
import pandas as pd
from constants import *
from data.MeteoDataset import MeteoDataset
import time

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
            locs[loc] = MeteoDataset(ts, df[col][2], df[col][3], df[col][1], df[col][0])
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

        t_df['block'] = [obj.block] * t_df.shape[0]
        t_df['district'] = [obj.district] * t_df.shape[0]
        t_df['latitude'] = [obj.latitude] * t_df.shape[0]
        t_df['longitude'] = [obj.longitude] * t_df.shape[0]

        p_df = pd.concat([p_df, t_df], ignore_index=True)
    
    return p_df

def create_meteo_dataframe(files, file_name, start_date, end_date):

    df = pd.DataFrame(columns=[x for x in METEO_COLUMNS_DICT.keys()])
    df = df.astype(METEO_COLUMNS_DICT)

    for f, start_index in files.items():

        start_time = time.time()

        f_df = pd.read_csv(f'{data_bihar}/{f}')
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

        if group.shape[0] != duration: continue
        locs.append(loc)

    df = df[df[['latitude', 'longitude']].apply(tuple, 1).isin(locs)]
    df.to_pickle(file_name, protocol=4)
    
    return df