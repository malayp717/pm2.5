import numpy as np
from shapely.geometry import Point, Polygon
from scipy.ndimage import gaussian_filter
from constants import *
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') 
from matplotlib import cm
from matplotlib.colors import Normalize, ListedColormap

def coordinate_bounds(region):
    coordinates_bound = region['geometry'].bounds
    min_lat, max_lat = coordinates_bound['miny'].min(), coordinates_bound['maxy'].max()
    min_long, max_long = coordinates_bound['minx'].min(), coordinates_bound['maxx'].max()
    return min_lat, max_lat, min_long, max_long

def get_indices(grid_long, grid_lat, region):
    points = [Point(lon, lat) for lon, lat in zip(grid_long.flatten(), grid_lat.flatten())]
    mask = np.zeros(len(points), dtype=bool)

    # Create a boolean mask to check if each point is within Bihar
    for geom in region.geometry:
        mask |= np.array([point.within(geom) for point in points])
    
    # Reshape the mask to match the shape of your original meshgrid
    mask_reshaped = mask.reshape(grid_lat.shape)

    return mask_reshaped

def favorable_points(grid_long, grid_lat, grid_values, region):
    mask = get_indices(grid_long, grid_lat, region)
    long, lat, values = grid_long[mask], grid_lat[mask], grid_values[mask]
    return np.array(long), np.array(lat), np.array(values)

def LCN(grid_long, grid_lat, grid_values):

    X, Y, normalized_values = np.copy(grid_long), np.copy(grid_lat), np.copy(grid_values)

    # Create a kernel for Gaussian smoothing
    sigma = 5  # Adjust this value based on your desired smoothing level
    kernel = np.exp(-0.5 * (X**2 + Y**2) / sigma**2)

    # Normalize the kernel
    kernel /= kernel.sum()
    # Apply Gaussian smoothing
    smoothed_values = gaussian_filter(normalized_values, sigma=sigma)

    return smoothed_values

def create_plot(data_long, data_lat, values, region, path, type):
    _, ax = plt.subplots(figsize=(50, 40))
    # ticks = [i for i in range(0, 550, 50)]
    vmin, vmax = values.min() if type == 'relative' else -50, values.max() if type == 'relative' else 1000
    norm = Normalize(vmin=vmin, vmax=vmax)

    my_cmap = cm.get_cmap('jet')
    my_cmap.set_over('darkred')
    my_cmap.set_under('blue')

    region.plot(ax=ax, color='white', edgecolor='grey', linewidth=0.5)
    scatter = ax.scatter(data_long, data_lat, c=values, cmap=my_cmap, marker='.', norm=norm)
    plt.colorbar(scatter, ax=ax, label=f'Predicted $PM_{2.5}$')
    # plt.colorbar(scatter, ax=ax, label='Predicted $PM_{2.5}$', orientation='vertical', extend='both', ticks=ticks)
    # cbar.set_clim(0, 1000)


    ax.set_axis_off()
    plt.savefig(f'{path}.jpg', dpi=400)
    plt.close()