import numpy as np
import matplotlib.pyplot as plt
import starfile
from scipy.spatial import cKDTree

# Dataframes contain .star files from template matching
# file_numbers usually refer to your Tomograms 1.rec, 2.rec,.., etc.
# For example like this: 

# file_numbers = ["1", "2", "3", "4"]
# list_path = ".../tomo_path/templateMatching/TM_bin4_"
# path = [list_path + file_number + ".star" for file_number in file_numbers]
# dataframes = {}
# for file_number, file_path in zip(file_numbers, path):
#    df = starfile.read(file_path)
#    dataframes[f"df{file_number}"] = df

# Initialize tree
trees = {}

# Set tomo pixel size and number of nearest neighbors (NN)
pixel_size = 7.84  # bin4 pixel size for example
k_values = range(2, 13)  # 12 NN's (can set k=2 for NN)

distances_dict = {}

for key, df in enumerate(dataframes):
    coords = df[['orig_x', 'orig_y', 'orig_z']].values * pixel_size
    tree = cKDTree(coords)
    distances, neighbors = tree.query(coords, k=max(k_values))
    distances_dict[key] = {f'distance_k{k}': distances[:, k-1] for k in k_values}
