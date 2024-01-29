import sys
sys.path.append('..')
import time
import pandas as pd
import numpy as np
from datetime import datetime
from constants import *
from eda_utils import *
import netCDF4 as nc
from netCDF4 import num2date
import xarray as xr
from shapely.geometry import Point, Polygon
import geopandas as gpd
import os
import logging
from PIL import Image
from itertools import product
import argparse

def log_list_content(log_file, my_list):
    # Configure logging to write to a file
    logging.basicConfig(filename=log_file, level=logging.DEBUG)

    # Log the content of the list
    logging.info("List content:")
    for item in my_list:
        logging.info(str(item))

def convert_era5_netcdf_to_dict(pbl_file, other_params_file, variable_names_pbl, variable_names_other):
    # Open the NetCDF file using xarray
    ds_pbl = xr.open_dataset(pbl_file)
    ds_other = xr.open_dataset(other_params_file)

    '''
        ******        List may contain lesser parameters        ******
            u10: 10 metre U wind component
            v10: 10 metre V wind component
            t2m: 2 metre temperature
            kx: K index
            sp: Surface pressure
            tp: Total precipitation
            blh: boundary layer height
    '''

    dataset = {'timestamp': [], 'latitude': [], 'longitude': [], 'blh': [], 'u10': [], 'v10': [], 'kx': [], 'sp': [], 'tp': []}

    # Extract data for the specified variable
    data = {}

    for var in variable_names_pbl:
        data[var] = ds_pbl[var].values

    for var in variable_names_other:
        data[var] = ds_other[var].values

    # Get time and spatial coordinates
    time = [pd.to_datetime(ts) for ts in ds_pbl['time'].values]
    lat = ds_pbl['latitude'].values
    lon = ds_pbl['longitude'].values

    # print(data)

    grid_lat, grid_lon = np.meshgrid(lat, lon)
    # Combine the grids into a 2D array
    points = [Point(lon, lat) for lon, lat in zip(grid_lon.flatten(), grid_lat.flatten())]
    mask_1d = np.zeros(len(points), dtype=bool)

    for geom in region.geometry:
        mask_1d |= np.array([point.within(geom) for point in points])

    grid = np.column_stack((grid_lon.ravel(), grid_lat.ravel()))
    locs = grid[mask_1d]

    mask_2d = mask_1d.reshape((lat.shape[0], -1))

    for i, var in enumerate(data):
        for ts, row in enumerate(data[var]):
            if i == 0:
                dataset['timestamp'].extend([time[ts]] * len(locs))
                dataset['longitude'].extend([x[0] for x in locs])
                dataset['latitude'].extend([x[1] for x in locs])

            dataset[var].extend(row[0][mask_2d])

    for var in variable_names_pbl:
        assert len(dataset[var]) == len(dataset['timestamp']) == len(dataset['longitude']) == len(dataset['latitude']), "Error in logic"
            
    for var in variable_names_other:
        assert len(dataset[var]) == len(dataset['timestamp']) == len(dataset['longitude']) == len(dataset['latitude']), "Error in logic"

    ds_pbl.close()
    ds_other.close()

    return dataset

def load_images(image_base_dir):

    locations = set()
    data = {'timestamp': [], 'block': [], 'district': [], 'image': []}

    for subdir, _, files in os.walk(image_base_dir):
        for file in files:
            #print os.path.join(subdir, file)
            filepath = subdir + os.sep + file

            if filepath.endswith(".tif"):

                # Extract location and date information from the file path
                info = [x.lower() for x in filepath.split('/')[-3:]]
                locations.add((info[0], info[1]))

                # Extract date information from the file_name
                date = info[2][:8]
                date_object = datetime.strptime(date, '%Y%m%d')
                timestamp = date_object.strftime('%Y-%m-%d %H:%M:%S')
                # print (timestamp, info)

                # Add all these parameters to the dataset
                data['district'].append(info[0])
                data['block'].append(info[1])
                data['timestamp'].append(timestamp)

                img = Image.open(filepath)
                img_array = np.array(img)
                data['image'].append(img_array)
            
    return data

if __name__ == '__main__':

    # Parser Arguments
    parser = argparse.ArgumentParser()

    # Define command-line arguments
    parser.add_argument('-e', '--era5', action='store_true', help='ERA5 data load')
    parser.add_argument('-i', '--img', action='store_true', help='Image data load')
    # parser.add_argument('-j', '--join', action='store_true', help='Join ERA5 data with Image data')
    parser.add_argument('-f', '--fill', action='store_true', help='Impute missing data using Iterative Imputer')
    parser.add_argument('-u', '--uts', action='store_true', help='Add NaN rows to make uniform shape across different locations')
    parser.add_argument('-r', '--rnn', action='store_true', help='Impute NaN rows for certain locations to make dataset RNN ready')

    # Parse the command-line arguments
    args = parser.parse_args()

    # All file paths and variables
    orig_data_file = f'{data_bihar}/bihar_512_sensor_data.pkl'
    pbl_file = f'{data_bihar}/PBLH_may_Dec_2023.nc'
    other_params_file = f'{data_bihar}/Era5_data_May_Dec_2023.nc'
    image_base_dir = f'{data_bihar}/Satellite_images/'
    geojson_file = f'{data_bihar}/bihar.json'
    log_file = f'./output.log'

    # output_params_file = f'{data_bihar}/params.pkl'
    output_era5_file = f'{data_bihar}/bihar_512_sensor_era5.pkl'
    # output_img_file = f'{data_bihar}/satellite_images.pkl'
    output_merged_file = f'{data_bihar}/bihar_512_sensor_era5_image.pkl'
    output_imputed_file = f'{data_bihar}/bihar_512_sensor_era5_image_imputed.pkl'
    output_uts_file = f'{data_bihar}/bihar_512_sensor_era5_image_uts.pkl'
    output_rnn_file = f'{data_bihar}/bihar_512_sensor_era5_rnn.pkl'

    variable_names_pbl = ['blh']
    variable_names_other = ['u10', 'v10', 'kx', 'sp', 'tp']

    orig_df = pd.read_pickle(orig_data_file)
    region = gpd.read_file(geojson_file)


    # If we want to make output_era5_file pickle
    if args.era5:

        start_time = time.time()
        print('******\t\tERA5 Data loading\t\t******')

        params_data = convert_era5_netcdf_to_dict(pbl_file, other_params_file, variable_names_pbl, variable_names_other)

        # # lower_limit = pd.to_datetime('2023-05-01 00:00:00')
        upper_limit = pd.to_datetime('2023-11-30 23:00:00')
        cols = {'timestamp': 'datetime64[ns]',  'latitude': np.float64, 'longitude': np.float64, 'blh': np.float64, 'u10': np.float64,\
                'v10': np.float64, 'kx': np.float64, 'sp': np.float64, 'tp': np.float64}
        
        params_df = pd.DataFrame(data=params_data, columns=cols)
        params_df = params_df[params_df['timestamp'] <= upper_limit]

        params_df['latitude'], params_df['longitude'] = transform_lat_long(orig_df, params_df)
        params_df = params_df.drop_duplicates(subset=['timestamp', 'latitude', 'longitude'])

        merged_df = pd.merge(orig_df, params_df, on=['timestamp', 'latitude', 'longitude'], how='left')
        merged_df = merged_df[['timestamp', 'block', 'district', 'latitude', 'longitude', 'rh', 'temp', 'blh', 'u10', 'v10', 'kx', 'sp',\
                               'tp', 'pm25']]

        merged_df.to_pickle(output_era5_file, protocol=4)

        print('******\t\tERA5 Data loading completed\t\t******')
        print(f'Time Elapsed:\t {(time.time() - start_time):.2f} s\n')

    # If we want to load image csv
    if args.img:
        
        start_time = time.time()
        print('******\t\tImage Data loading\t\t******')

        img_df = pd.DataFrame(data=load_images(image_base_dir))
        img_df['timestamp'] = img_df['timestamp'].astype('datetime64[ns]')
        img_df = img_df.sort_values(by='timestamp')

        img_df = img_df.drop_duplicates(subset=['timestamp', 'block', 'district'])
        # duplicate_count_specific = img_df.duplicated(subset=['timestamp', 'block', 'district']).sum()
        # print(duplicate_count_specific)

        # img_df.to_pickle(output_img_file)

        era5_df = pd.read_pickle(output_era5_file)
        era5_df['block'] = era5_df['block'].str.lower()
        era5_df['district'] = era5_df['district'].str.lower()

        merged_df = pd.merge(era5_df, img_df, on=['timestamp', 'block', 'district'], how='left')

        merged_df = merged_df[['timestamp', 'block', 'district', 'latitude', 'longitude', 'rh', 'temp', 'blh', 'u10', 'v10', 'kx', 'sp',\
                            'tp', 'image', 'pm25']]
        
        # cnt = merged_df['image'].isna().sum()
        # print(f'NaN Count: {cnt}')
        # print(merged_df.shape, img_df.shape, era5_df.shape)
        # print(era5_df.columns)
        # print(merged_df.columns)

        merged_df.to_pickle(output_merged_file, protocol=4)

        print('******\t\tImage Data loading completed\t\t******')
        print(f'Time Elapsed:\t {(time.time() - start_time):.2f} s\n')

    if args.fill:

        merged_df = pd.read_pickle(output_merged_file)
        merged_df['timestamp'] = merged_df['timestamp'].astype('datetime64[ns]')

        orig_data = merged_df.copy(deep=True)[['timestamp', 'latitude', 'longitude', 'rh', 'temp', 'blh', 'u10', 'v10', 'kx', 'sp', 'tp', 'pm25']]
        orig_data['timestamp'] = orig_data['timestamp'].values.astype(float)

        orig_data = orig_data.to_numpy()

        print('******\t\tImputing Data\t\t******')
        start_time = time.time()

        imputed_data = impute(orig_data)
        # print(imputed_data.shape)

        cols = {'timestamp': 'datetime64[ns]', 'block': str, 'district': str, 'latitude': np.float64, 'longitude': np.float64,\
                'rh': np.float64, 'temp': np.float64, 'blh': np.float64, 'u10': np.float64, 'v10': np.float64,\
                'kx': np.float64, 'sp': np.float64, 'tp': np.float64, 'image': np.ndarray, 'pm25': np.float64}
    
        f_cols = {x: y for x, y in cols.items() if x not in {'timestamp', 'block', 'district', 'image'}}
        # print(f_cols)

        imputed_df = pd.DataFrame(imputed_data[:,1:], columns=f_cols)
        imputed_df[['timestamp', 'block', 'district', 'image']] = merged_df[['timestamp', 'block', 'district', 'image']]
        imputed_df = imputed_df[[x for x in cols.keys()]]

        imputed_df.to_pickle(output_imputed_file, protocol=4)

        print('******\t\tImputing Data Completed\t\t******')
        print(f'Time Elapsed:\t {(time.time() - start_time):.2f} s\n')
    
    if args.uts:

        print('******\t\tUniform Shape processing start\t\t******')
        start_time = time.time()

        f_df = pd.read_pickle(output_imputed_file)
        loc_info = f_df.groupby(['latitude', 'longitude', 'block', 'district']).groups.keys()

        all_timestamps = f_df['timestamp'].unique()
        all_locations = f_df.groupby(['latitude', 'longitude', 'block', 'district']).groups.keys()
        all_combinations = list(product(all_timestamps, all_locations))

        len_ts, len_l, len_c = len(all_timestamps), len(all_locations), len(all_combinations)
        # print(len_ts, len_l, len_c, len_ts * len_l)

        new_cols = {'timestamp': 'datetime64[ns]', 'locations': tuple}
        new_df = pd.DataFrame(data=all_combinations, columns=new_cols)

        new_df['latitude'] = new_df['locations'].apply(lambda x : x[0])
        new_df['longitude'] = new_df['locations'].apply(lambda x : x[1])
        new_df['block'] = new_df['locations'].apply(lambda x : x[2])
        new_df['district'] = new_df['locations'].apply(lambda x : x[3])
        new_df = new_df[['timestamp', 'latitude', 'longitude', 'block', 'district']]

        result_df = pd.merge(new_df, f_df, how='left')
        df_grouped = result_df.groupby(['latitude', 'longitude', 'block', 'district'])

        for loc, group in df_grouped:
            assert group.shape[0] == len_ts
        
        result_df.to_pickle(output_uts_file, protocol=4)

        print('******\t\tUniform Shape processing Completed\t\t******')
        print(f'Time Elapsed:\t {(time.time() - start_time):.2f} s\n')

    if args.rnn:
        print('******\t\tRNN data processing start\t\t******')
        start_time = time.time()

        uts_df = pd.read_pickle(output_uts_file)
        uts_df = uts_df[[col for col in COLUMNS_DICT.keys()]]
        uts_df_grouped = uts_df.groupby(['latitude', 'longitude'])

        rnn_df = pd.DataFrame(columns=COLUMNS_DICT)


        for loc, group in uts_df_grouped:
            if group['pm25'].count() >= THRESHOLD:

                group_filled = group.interpolate(method='linear', axis=0, limit_direction='both')
                rnn_df = pd.concat([rnn_df, group_filled], ignore_index=True)
        
        rnn_df.sort_values(by='timestamp', inplace=True)
        rnn_df.to_pickle(output_rnn_file, protocol=4)

        print('******\t\tRNN data processing completed\t\t******')
        print(f'Time Elapsed:\t {(time.time() - start_time):.2f} s\n')