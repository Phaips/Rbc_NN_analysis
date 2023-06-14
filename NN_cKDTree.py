import numpy as np
import matplotlib.pyplot as plt
import starfile
from scipy.spatial import cKDTree

# Dataframes contain .star files from template matching as df

trees = {}
pixel_size = 7.84  # bin4 pixel size
k_values = range(2, 13)  # 12 NN
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


# Angular analysis

dfcos = {}

for key, df in enumerate(dataframes):
    cos_values = []
    for i, neighbors_indices in enumerate(neighbors):
        particle_angles = df.loc[[i], ['phi', 'psi', 'the']].values
        neighbor_angles = df.loc[neighbors_indices[1:], ['phi', 'psi', 'the']].values  # Exclude the particle itself
        cos_angles = np.cos(np.deg2rad(neighbor_angles - particle_angles))
        cos_values.append(cos_angles)
    dfcos[key] = cos_values

