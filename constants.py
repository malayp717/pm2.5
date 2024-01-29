import numpy as np

base_dir = '/home/malay/pm25/pm2.5'
data_dir = f'/hdd/malay'
data_bihar = f'{data_dir}/bihar'
model_dir = f'{data_dir}/models'
plot_dir = f'{data_dir}/plots'
bihar_plot_dir = f'{plot_dir}/bihar'
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
COLUMNS_DICT = {'timestamp': 'datetime64[ns]', 'latitude': np.float64, 'longitude': np.float64, 'rh': np.float64,\
                'temp': np.float64,'blh': np.float64, 'u10': np.float64, 'v10': np.float64, 'kx': np.float64, 'sp': np.float64,\
                    'tp': np.float64, 'pm25': np.float64}
THRESHOLD = 9_000
WINDOW_SIZE = 72
GRID_SIZE = 1_000
AREA_BIHAR = 94_163