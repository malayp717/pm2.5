import os
import yaml
import numpy as np
import pandas as pd
# from utils import eval_stat
import faiss
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(f'{proj_dir}/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# ------------- Config parameters start ------------- #
meteo_var = config['meteo_var']
data_dir = config['dirpath']['data_dir']
# ------------- Config parameters end ------------- #

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

def npy_data(df, locations_fp, npy_fp):
    locations = pd.read_csv(locations_fp, sep='|', header=None)
    locs_grouped = df.groupby(['longitude', 'latitude'])

    locs_to_index_dict = {}

    for idx, row in locations.iterrows():
        locs_to_index_dict[(row[3], row[4])] = row[0] 

    T, L, F = len(list(df['timestamp'].unique())), locations.shape[0], df.shape[-1]-3
    npy_data = np.zeros((T, L, F))

    for loc, group in locs_grouped:
        group = group.sort_values(by='timestamp')
        l = locs_to_index_dict[loc]

        for t in range(T):
            npy_data[t][l] = group.iloc[t][-9:]

    with open(npy_fp, 'wb') as f:
        np.save(f, npy_data)

def impute_data(df, output_meteo_era5_imputed_file, method):
    df['timestamp'] = df['timestamp'].astype('datetime64[ns]')

    orig_data = df.copy(deep=True)
    orig_data['timestamp'] = orig_data['timestamp'].values.astype(float)

    orig_data = orig_data.to_numpy()
    imputed_data = impute(orig_data, method=method)

    imputed_df = pd.DataFrame(imputed_data, columns=df.columns)
    imputed_df['timestamp'] = pd.to_datetime(imputed_df['timestamp'])

    df = imputed_df.copy(deep=True)
    df = df[meteo_var]
    df['rh'] = df['rh'].clip(lower=0)
    df['pm25'] = df['pm25'].clip(lower=0)
    df.to_pickle(output_meteo_era5_imputed_file, protocol=4)

    del imputed_df
    del imputed_data
    del orig_data

    return df