import numpy as np
from shapely.geometry import Point, Polygon
from scipy.ndimage import gaussian_filter
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

def get_indices(grid_long, grid_lat, region, mask_fp):
    points = [Point(lon, lat) for lon, lat in zip(grid_long.flatten(), grid_lat.flatten())]
    mask = np.zeros(len(points), dtype=bool)

    # Create a boolean mask to check if each point is within Bihar
    for geom in region.geometry:
        mask |= np.array([point.within(geom) for point in points])
    
    # Reshape the mask to match the shape of your original meshgrid
    mask_reshaped = mask.reshape(grid_lat.shape)

    np.savetxt(mask_fp, mask_reshaped, fmt='%5i', delimiter='\t')
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
    _, ax = plt.subplots(figsize=(8, 6))
    vmin, vmax = values.min() if type == 'relative' else 0, values.max() if type == 'relative' else 300
    norm = Normalize(vmin=vmin, vmax=vmax)

    my_cmap = cm.get_cmap('jet')
    my_cmap.set_over('darkred')
    my_cmap.set_under('blue')

    region.plot(ax=ax, color='white', edgecolor='grey', linewidth=0.5)
    scatter = ax.scatter(data_long, data_lat, c=values, cmap=my_cmap, marker='.', norm=norm)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(f'$PM_{2.5}$ Concentration', fontsize=12)

    ax.set_axis_off()
    plt.savefig(f'{path}.jpg')
    plt.close()