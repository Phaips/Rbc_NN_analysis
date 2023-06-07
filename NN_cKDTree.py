import numpy as np
import matplotlib.pyplot as plt
import starfile
from scipy.spatial import cKDTree

# Dataframes contain .star files from template matching as df with x,y,z coords. 

trees = {}
pixel_size = 7.84 # bin4 pixel size
k_values = range(2, 14) # 12 NN
distances_dict = {}

for key, df in dataframes.items():
    coords = df[['orig_x', 'orig_y', 'orig_z']].values * pixel_size
    tree = cKDTree(coords)
    distances_dict[key] = {}

    for k in k_values:
        distances, neighbors = tree.query(coords, k=k)
        distances_dict[key][f'distance_k{k}'] = distances[:, k-1]  # Distances for k value

mean_distances = {}
median_distances = {}

for key, distances in distances_dict.items():
    mean_distances[key] = {}
    median_distances[key] = {}

    for k, dist in distances.items():
        mean_distances[key][k] = np.mean(dist)
        median_distances[key][k] = np.median(dist)
