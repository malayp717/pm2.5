import numpy as np
import xarray as xr
import pandas as pd
from shapely.geometry import Point
from data.eda_utils import transform_lat_long

def convert_era5_netcdf_to_dict(region, pbl_file, other_params_file, variable_names_pbl, variable_names_other):
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

def create_era5_dataframe(region, meteo_df, pbl_file, other_params_file, variable_names_pbl, variable_names_other, start_date, end_date,\
                   output_meteo_era5_file):
    params_data = convert_era5_netcdf_to_dict(region, pbl_file, other_params_file, variable_names_pbl, variable_names_other)

    params_cols = {'timestamp': 'datetime64[ns]',  'latitude': np.float64, 'longitude': np.float64, 'blh': np.float64, 'u10': np.float64,\
            'v10': np.float64, 'kx': np.float64, 'sp': np.float64, 'tp': np.float64}

    params_df = pd.DataFrame(data=params_data, columns=params_cols)
    params_df = params_df[params_df['timestamp'] < end_date]

    params_df['latitude'], params_df['longitude'] = transform_lat_long(meteo_df, params_df)
    params_df = params_df.drop_duplicates(subset=['timestamp', 'latitude', 'longitude'])

    merged_df = pd.merge(meteo_df, params_df, on=['timestamp', 'latitude', 'longitude'], how='left')
    merged_df = merged_df[['timestamp', 'block', 'district', 'latitude', 'longitude', 'rh', 'temp', 'blh', 'u10', 'v10', 'kx', 'sp',\
                            'tp', 'pm25']]

    merged_df.to_pickle(output_meteo_era5_file, protocol=4)
    return merged_df