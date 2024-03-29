import numpy as np
import pandas as pd
# from utils import eval_stat
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler

''' Use inbuilt sklearn functions to fill the missing nan values in the data '''
def impute(data, method='iterative'):
    assert method == 'knn' or method == 'mean' or method == 'iterative', 'method can only knn, mean or iterative'
    # KNN Imputer

    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)

    if method == 'knn':
        imputer = KNNImputer(n_neighbors=3)
    # Mean Imputer
    elif method == 'mean':
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    # Iterative Imputer
    elif method == 'iterative':
        imputer = IterativeImputer(random_state=0)
    
    imputed_data = imputer.fit_transform(normalized_data)
    return scaler.inverse_transform(imputed_data)

def find_distance(x, y):
    return np.abs(x[0] - y[0]) + np.abs(x[1] - y[1])

''' ERA5 file resolution is different to the original dataset resolution, find the closest location and return
    Parameters:
        orig_df: Original Dataframe
        params_df: ERA5 dataframe
'''
def transform_lat_long(orig_df, params_df):
    lat_lon_orig = list(orig_df.groupby(['latitude', 'longitude']).groups.keys())
    lat_lon_params = list(params_df.groupby(['latitude', 'longitude']).groups.keys())

    # Lat_Lon in the NetCDRF file -> Lat_Lon in the original dataset
    close_locs = {}

    for loc_params in lat_lon_params:
        dist = 10**9
        for loc_orig in lat_lon_orig:
            if find_distance(loc_orig, loc_params) < dist:
                dist = find_distance(loc_orig, loc_params)
                close_locs[loc_params] = loc_orig

    transformed_locs = [close_locs[(x, y)] for (x, y) in zip(params_df['latitude'], params_df['longitude'])]
    
    return map(list, zip(*transformed_locs))

def impute_data(df, output_meteo_era5_imputed_file, method):
    df['timestamp'] = df['timestamp'].astype('datetime64[ns]')

    orig_data = df.copy(deep=True)[['timestamp', 'latitude', 'longitude', 'rh', 'temp', 'blh', 'u10', 'v10', 'kx', 'sp', 'tp', 'pm25']]
    orig_data['timestamp'] = orig_data['timestamp'].values.astype(float)

    orig_data = orig_data.to_numpy()

    imputed_data = impute(orig_data, method=method)
    # print(imputed_data.shape)

    cols = {'timestamp': 'datetime64[ns]', 'block': str, 'district': str, 'latitude': np.float64, 'longitude': np.float64,\
            'rh': np.float64, 'temp': np.float64, 'blh': np.float64, 'u10': np.float64, 'v10': np.float64,\
            'kx': np.float64, 'sp': np.float64, 'tp': np.float64, 'pm25': np.float64}

    f_cols = {x: y for x, y in cols.items() if x not in {'timestamp', 'block', 'district'}}

    imputed_df = pd.DataFrame(imputed_data[:,1:], columns=f_cols)
    imputed_df[['timestamp', 'block', 'district']] = df[['timestamp', 'block', 'district']]
    imputed_df = imputed_df[[x for x in cols.keys()]]

    df = imputed_df.copy(deep=True)
    df.to_pickle(output_meteo_era5_imputed_file, protocol=4)

    del imputed_df
    del imputed_data
    del orig_data

    return df