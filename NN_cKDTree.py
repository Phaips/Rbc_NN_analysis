import numpy as np
import matplotlib.pyplot as plt
import starfile
from scipy.spatial import cKDTree

# Dataframes contain .star files from template matching as df with x,y,z coords. 

trees = {}
pixel_size = 7.84  # bin4 pixel size
k_values = range(2, 14)  # 12 NN
distances_dict = {}
mean_distances = {}
median_distances = {}

for key, df in enumerate(dataframes):
    coords = df[['orig_x', 'orig_y', 'orig_z']].values * pixel_size
    tree = cKDTree(coords)
    distances, neighbors = tree.query(coords, k=max(k_values))
    distances_dict[key] = {f'distance_k{k}': distances[:, k-1] for k in k_values}

    mean_distances[key] = {k: np.mean(distances_dict[key][f'distance_k{k}']) for k in k_values}
    median_distances[key] = {k: np.median(distances_dict[key][f'distance_k{k}']) for k in k_values}
