import numpy as np
from scipy.spatial import distance
import starfile

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

pixel_size = 7.84
nearest_distances = {}
for df_name, df in dataframes.items():
    coordinates = df[['orig_x', 'orig_y', 'orig_z']].values * pixel_size
    # Calculate pairwise distances
    dist_matrix = distance.cdist(coordinates, coordinates)
    # Set diagonal values to infinity to exclude self-distances
    np.fill_diagonal(dist_matrix, np.inf)
    # Calculate the nearest distance for each particle
    nearest_dist = np.min(dist_matrix, axis=1)
    nearest_distances[df_name] = nearest_dist
