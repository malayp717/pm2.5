import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import geopandas as gpd
import argparse
from constants import *
from data.load_meteo_data import create_meteo_dataframe
from data.load_era5_data import create_era5_dataframe
from data.eda_utils import impute_data
from pathlib import Path

if __name__ == '__main__':

    # parser = argparse.ArgumentParser()

    # # Define command-line arguments
    # parser.add_argument('-e', '--era5', action='store_true', help='ERA5 data load')
    # parser.add_argument('-i', '--img', action='store_true', help='Image data load')
    # # parser.add_argument('-j', '--join', action='store_true', help='Join ERA5 data with Image data')
    # parser.add_argument('-f', '--fill', action='store_true', help='Impute missing data using Iterative Imputer')
    # parser.add_argument('-u', '--uts', action='store_true', help='Add NaN rows to make uniform shape across different locations')
    # parser.add_argument('-r', '--rnn', action='store_true', help='Impute NaN rows for certain locations to make dataset RNN ready')

    # # Parse the command-line arguments
    # args = parser.parse_args()
    
    '''----------------- All the file locations are declared here -----------------'''

    pbl_file = f'{data_bihar}/PBLH_may_Dec_2023.nc'
    other_params_file = f'{data_bihar}/Era5_data_May_Dec_2023.nc'
    geojson_file = f'{data_bihar}/bihar.json'
    out_meteo_file = f'{data_bihar}/bihar_meteo_may_jan.pkl'
    output_meteo_era5_file = f'{data_bihar}/bihar_meteo_era5_may_jan.pkl'
    output_meteo_era5_imputed_file = f'{data_bihar}/bihar_meteo_era5_may_jan_iterative_imputed.pkl'

    region = gpd.read_file(geojson_file)

    '''----------------- file locations declaration complete -----------------'''


    '''----------------- All the variables are declared here -----------------'''

    start_date, end_date = pd.Timestamp('2023-05-01 00:00:00'), pd.Timestamp('2024-02-01 00:00:00')

    # Mapping: File Name to the corresponding row in which the timestamp information starts
    files = {'Bihar_536_Sensor_Data_Sep_2023_Screened.csv': 7, 'Bihar_536_Sensor_Data_Oct_2023_Screened.csv': 7,\
        'Bihar_536_Sensor_Data_Nov_2023_Screened.csv': 6, 'Bihar_536_Sensor_Data_Jan_2024_Screened.csv': 6,\
        'Bihar_536_Sensor_Data_Dec_2023_Screened.csv': 6, 'Bihar_512_Sensor_Data_May_Aug_Screened_Hourly.csv': 6}
    
    params_files = ['era5_may_dec_2023.nc', 'era5_jan_2024.nc']

    '''----------------- Variables declaration complete -----------------'''


    '''----------------- Meteo Data Load start -----------------'''

    start_time = time.time()
    print('******\t\tMeteo Data loading\t\t******')
    
    if Path(out_meteo_file).is_file():
        df = pd.read_pickle(out_meteo_file)
    else:
        df = create_meteo_dataframe(files, out_meteo_file, start_date, end_date)

    print('******\t\tMeteo Data loading completed\t\t******')
    print(f'Time Elapsed:\t {(time.time() - start_time):.2f} s\n')

    '''----------------- Meteo Data Load end -----------------'''

    '''----------------- ERA5 Data Load start -----------------'''

    start_time = time.time()
    print('******\t\tERA5 Data loading\t\t******')

    if Path(output_meteo_era5_file).is_file():
        df = pd.read_pickle(output_meteo_era5_file)
    else:
        df = create_era5_dataframe(params_files, df, output_meteo_era5_file)
    
    print('******\t\tERA5 Data loading completed\t\t******')
    print(f'Time Elapsed:\t {(time.time() - start_time):.2f} s\n')

    '''----------------- ERA5 Data Load end -----------------'''

    print(df.count())
    for col, type in zip(df.columns, df.dtypes):
        if type == np.float64 or type == np.float32: print(col, df[col].min(), df[col].max())
    
    '''----------------- Data Imputation start -----------------'''

    start_time = time.time()
    print('******\t\tData Imputation starting\t\t******')

    if Path(output_meteo_era5_imputed_file).is_file():
        df = pd.read_pickle(output_meteo_era5_imputed_file)
    else:
        df = impute_data(df, output_meteo_era5_imputed_file, method='iterative')
    
    print('******\t\tData Imputation completed\t\t******')
    print(f'Time Elapsed:\t {(time.time() - start_time):.2f} s\n')

    '''----------------- Data Imputation end -----------------'''

    print(df.count())
    for col, type in zip(df.columns, df.dtypes):
        if type == np.float64 or type == np.float32: print(col, df[col].min(), df[col].max())