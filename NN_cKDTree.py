import numpy as np
import matplotlib.pyplot as plt
import starfile
from scipy.spatial import cKDTree

# pixelsize of your tomo
pixel_size = 7.84
# number of nearest neighbors (k=2 here)
k_values = range(2, 3) 

# load your motl (.star file)
path = "../your/path/motl.star"
tomo = starfile.read(path)

# initialize df and sort by tomo number
dataframes = {}
unique_tomo_nums = sorted(tomo['tomo_num'].unique())    
for tomo_num in unique_tomo_nums:
    dataframes[f"df{tomo_num}"] = tomo[tomo['tomo_num'] == tomo_num].reset_index(drop=True)

# calculate the distance between two angles
def angular_difference(angle1, angle2):
    diff = np.abs(angle1 - angle2) % 360 # apparently STOPGAP phi angles are not mod 360 (??)
    wrap = np.minimum(diff, 360 - diff) # if 10° and 350° you want 20° and not -340° (D4 symmetry sucks)
    return(wrap)
    
# you have 3 angles, this function allows to chose index 0,1,2 for phi,psi,theta respectively
def calculate_angle_diffs(euler_angles, neighbor_angles, angle_index):
    return angular_difference(euler_angles[:, angle_index], neighbor_angles[:, angle_index])

# 0 for phi, 1 for psi, 2 for theta
angle_index = 2 

#initialize dicts
angles_dict = {}
distances_dict = {}

for key, df in dataframes.items():
    coords = df[['orig_x', 'orig_y', 'orig_z']].values * pixel_size
    euler_angles = df[['phi', 'psi', 'the']].values
    # spate partitioning to find nearest neighbors
    tree = cKDTree(coords)
    distances, neighbors = tree.query(coords, k=max(k_values))
    
    angles_dict[key] = {}
    distances_dict[key] = {}
    for k in k_values:
        # NN distance
        distances_dict[key][f'distance_k{k}'] = distances[:, k-1]
        # find angle of nearest neighbors
        neighbor_indices = neighbors[:, k - 1]
        neighbor_angles = euler_angles[neighbor_indices]
        #calculate their difference
        angle_diffs = calculate_angle_diffs(euler_angles, neighbor_angles, angle_index)
        angles_dict[key][f'angle_diff_{angle_index}_k{k}'] = angle_diffs



"""
# PLOT THE NN DISTANCES AND ANGLE DIFFERENCE

colormap_name = 'tab10'
colormap = plt.get_cmap(colormap_name)

num_colors_needed = len(k_values)
colors = [colormap(i) for i in range(num_colors_needed)]

import seaborn as sns
for i, (key, df) in enumerate(dataframes.items()):
    # Create a 1x4 grid of subplots (1 row, 4 columns) for each key
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))  # Adjust figsize if needed

    # Plot the density plot for distances and overlay for different k values (Improved Visualization)
    ax_distance = axs[0]
    for j, k in enumerate(k_values):
        color = colors[j]
        sns.kdeplot(
            distances_dict[key][f'distance_k{k}'], 
            color=color, label=f'k={k}', ax=ax_distance
        )
    ax_distance.set_xlabel('Distance to Nearest Neighbor (Å)')
    ax_distance.set_ylabel('Density')
    ax_distance.set_title(f'{key} - Distance')
    ax_distance.legend(loc='upper right')

    # Plot the density plot for each angle index and overlay the angle differences for different k values
    for angle_idx in range(3):  # Loop through angle indices 0, 1, and 2
        ax = axs[angle_idx + 1]  # Select the appropriate subplot for the angle index
        ax.set_xlabel('Angular Difference')
        ax.set_ylabel('Density')
        ax.set_title(f'{key} - {["Phi", "Psi", "Theta"][angle_idx]} Differences')
        
        for j, k in enumerate(k_values):
            color = colors[j]
            sns.kdeplot(
                angles_dict[key][f'angle_diff_{angle_idx}_k{k}'], 
                color=color, label=f'k={k}', ax=ax
            )
        
        ax.legend(loc='upper right')

    # Adjust spacing and layout
    plt.tight_layout()
    plt.show()

"""


