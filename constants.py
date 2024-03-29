base_dir = '/home/malay/pm25/pm2.5'
data_dir = f'/hdd/malay/'
data_bihar = f'{data_dir}/bihar/'
model_dir = f'{data_dir}/models/'
plot_dir = f'{data_dir}/plots/'
bihar_plot_dir = f'{plot_dir}/bihar/'

METEO_COLUMNS_DICT = {'timestamp': 'datetime64[ns]', 'district': object, 'block': object, 'latitude': float, 'longitude': float,\
                                 'rh': float, 'temp': float, 'pm25': float}
COLUMNS_DICT = {'timestamp': 'datetime64[ns]', 'latitude': float, 'longitude': float, 'rh': float, 'temp': float,'blh': float, 'u10': float,\
                'v10': float, 'kx': float, 'sp': float, 'tp': float, 'pm25': float}
THRESHOLD = 9_000
LOWER_BOUND = 0
UPPER_BOUND = 600
GRID_SIZE = 1_000
AREA_BIHAR = 94_163