import os
import yaml
import numpy as np
import xarray as xr
import pandas as pd
from shapely.geometry import Point
from data.eda_utils import transform_lat_long
from scipy.spatial.distance import cdist
from haversine import haversine, Unit

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(f'{proj_dir}/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# ------------- Config parameters start ------------- #
location = 'bihar'
meteo_var = config[location]['meteo_var']
data_dir = config['dirpath']['data_dir']

METEO_COLUMNS_DICT = {x: float for x in meteo_var}
METEO_COLUMNS_DICT.update({
    'timestamp': 'datetime64[ns]'
})
# ------------- Config parameters end ------------- #

def find_closest_neighbors(locs_params, locs_meteo):
    locs_params = np.array(locs_params)
    locs_meteo = np.array(locs_meteo)
    
    # Calculate the distances between each pair of points in locs_params and locs_meteo
    distances = cdist(locs_params, locs_meteo, metric=(lambda x, y: haversine(x, y, unit=Unit.KILOMETERS)))
    
    # Find the indices of the closest neighbors in locs_params
    closest_neighbor_indices = np.argmin(distances, axis=1)
    
    # Get the coordinates of the closest neighbors from locs_params
    closest_neighbors = locs_params[closest_neighbor_indices]
    return closest_neighbors

def create_era5_dataframe(nc_files, meteo_df, output_meteo_era5_file):

    params_cols = {'timestamp': 'datetime64[ns]',  'latitude': np.float64, 'longitude': np.float64, 'blh': np.float64, 'u10': np.float64,\
            'v10': np.float64, 'kx': np.float64, 'sp': np.float64, 'tp': np.float64}
    
    era5_df = pd.DataFrame(columns=params_cols.keys())
    era5_df = era5_df.astype(params_cols)

    for ncf in nc_files:
        ds = xr.open_dataset(f'{data_dir}/{ncf}')

        # for var in ds.variables:
        #     var_info = ds[var]

        #     print("Variable Name:", var)
        #     for attr_name, attr_value in var_info.attrs.items():
        #         print(f"  {attr_name}: {attr_value}")

        # lat, lon = ds.latitude.values, ds.longitude.values

        curr_df = ds.to_dataframe().reset_index()
        curr_df = curr_df.rename(columns={'time': 'timestamp'})
        curr_df = curr_df.sort_values(by='timestamp').reset_index()
        curr_df = curr_df[params_cols.keys()]

        era5_df = pd.concat([era5_df, curr_df], ignore_index=True)

    era5_df_g = era5_df.groupby(['longitude', 'latitude'])
    era5_df_list = []

    for loc, grp in era5_df_g:
        grp['timestamp'] = grp['timestamp'].dt.floor('H')
        df_t = grp.groupby('timestamp').mean()
        df_t = df_t.reset_index()

        era5_df_list.append(df_t)
    
    era5_df = pd.concat(era5_df_list)
    era5_df_g = era5_df.groupby(['longitude', 'latitude'])

    locs_meteo = [tuple(row) for row in meteo_df[['longitude', 'latitude']].drop_duplicates().values]
    locs_params = [tuple(row) for row in era5_df[['longitude', 'latitude']].drop_duplicates().values]

    nbrs = find_closest_neighbors(locs_params, locs_meteo)
    nbrs_mapping = {loc: nbr for loc, nbr in zip(locs_meteo, nbrs)}

    era5_locs = []
    for loc, nbr in nbrs_mapping.items():
        loc, nbr = tuple(loc), tuple(nbr)
        nbrs_mapping[loc] = nbr
        era5_locs.append(nbr)

    # print(nbrs_mapping)

    era5_df['locs'] = era5_df[['longitude', 'latitude']].apply(tuple, 1)
    era5_df = era5_df[era5_df['locs'].isin(era5_locs)]
    era5_df = era5_df[[x for x in era5_df.columns if x not in ('longitude', 'latitude')]]

    meteo_df['locs'] = meteo_df[['longitude', 'latitude']].apply(tuple, 1)
    meteo_df_g = meteo_df.groupby('locs')
    era5_df_g = era5_df.groupby('locs')

    pd_list = []
    for loc, m_df in meteo_df_g:
        p_df = era5_df[era5_df['locs'] == nbrs_mapping[loc]]

        df_t = pd.merge(m_df, p_df, on='timestamp', how='left')
        df_t = df_t.sort_values(by='timestamp')

        pd_list.append(df_t)
    
    df = pd.concat(pd_list)
    df = df[meteo_var]
    df = df.sort_values(by='timestamp').reset_index()
    
    df.to_pickle(output_meteo_era5_file, protocol=4)
    return df