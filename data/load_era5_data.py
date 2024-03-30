import numpy as np
import xarray as xr
import pandas as pd
from shapely.geometry import Point
from data.eda_utils import transform_lat_long
from scipy.spatial.distance import cdist
from geopy.distance import geodesic
from constants import *

def find_closest_neighbors(locs_params, locs_meteo):
    locs_params = np.array(locs_params)
    locs_meteo = np.array(locs_meteo)
    
    # Calculate the distances between each pair of points in locs_params and locs_meteo
    distances = cdist(locs_params, locs_meteo, lambda x, y: geodesic(x, y).km)
    
    # Find the indices of the closest neighbors in locs_params
    closest_neighbor_indices = np.argmin(distances, axis=1)
    
    # Get the coordinates of the closest neighbors from locs_params
    closest_neighbors = locs_params[closest_neighbor_indices]
    return closest_neighbors

def create_era5_dataframe(nc_files, meteo_df, output_meteo_era5_file):

    params_cols = {'timestamp': 'datetime64[ns]',  'latitude': np.float64, 'longitude': np.float64, 'blh': np.float64, 'u10': np.float64,\
            'v10': np.float64, 'kx': np.float64, 'sp': np.float64, 'tp': np.float64}
    
    params_df = pd.DataFrame(columns=params_cols.keys())
    params_df = params_df.astype(params_cols)

    for ncf in nc_files:
        ds = xr.open_dataset(f'{data_bihar}/{ncf}')

        # for var in ds.variables:
        #     if var in cols:
        #         var_info = ds[var]

        #         print("Variable Name:", var)
        #         for attr_name, attr_value in var_info.attrs.items():
        #             print(f"  {attr_name}: {attr_value}")

        # lat, lon = ds.latitude.values, ds.longitude.values

        curr_df = ds.to_dataframe().reset_index()
        curr_df = curr_df.rename(columns={'time': 'timestamp'})
        curr_df = curr_df.sort_values(by='timestamp').reset_index()
        curr_df = curr_df[params_cols.keys()]

        params_df = pd.concat([params_df, curr_df], ignore_index=True)

    locs_meteo = [tuple(row) for row in meteo_df[['latitude', 'longitude']].drop_duplicates().values]
    locs_params = [tuple(row) for row in params_df[['latitude', 'longitude']].drop_duplicates().values]

    nbrs = find_closest_neighbors(locs_params, locs_meteo)
    nbrs_mapping = {loc: nbr for loc, nbr in zip(locs_meteo, nbrs)}

    for loc, nbr in nbrs_mapping.items():
        nbrs_mapping[loc] = tuple(nbr)

    # print(nbrs_mapping)

    params_df['locs'] = params_df[['latitude', 'longitude']].apply(tuple, 1)
    params_df = params_df[[x for x in params_df.columns if x not in ('latitude', 'longitude')]]

    meteo_df['locs'] = meteo_df[['latitude', 'longitude']].apply(tuple, 1).map(nbrs_mapping)

    df = pd.merge(meteo_df, params_df, on=['timestamp', 'locs'], how='left')
    df = df[['timestamp', 'block', 'district', 'latitude', 'longitude', 'rh', 'temp', 'blh', 'u10', 'v10', 'kx', 'sp',\
                'tp', 'pm25']]

    df = df.sort_values(by='timestamp').reset_index()
    
    df.to_pickle(output_meteo_era5_file, protocol=4)
    return df