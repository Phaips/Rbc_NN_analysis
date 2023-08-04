import numpy as np
import matplotlib.pyplot as plt
import starfile
from scipy.spatial import cKDTree
import seaborn as sns

# pixelsize of your tomo
pixel_size = 7.84
# number of nearest neighbors (k=2 here)
k_values = range(2, 4) 

# load your motl (.star file)
path = "../your/path/motl.star"
tomo = starfile.read(path)

# initialize df and sort by tomo number
dataframes = {f"df{tomo_num}": df.reset_index(drop=True) for tomo_num, df in tomo.groupby('tomo_num')}

# calculate the distance between two angles
def angular_difference(angle1, angle2):
    diff = np.abs(angle1 - angle2) % 360
    wrap = np.minimum(diff, 360 - diff)
    return wrap

# you have 3 angles, this function allows to chose index 0,1,2 for phi,psi,theta respectively
def calculate_angle_diffs(euler_angles, neighbor_angles, angle_index):
    return angular_difference(euler_angles[:, angle_index], neighbor_angles[:, angle_index])

# 0 for phi, 1 for psi, 2 for theta
angle_index = 2 

# Initialize dicts using list comprehensions
angles_dict = {key: {f'angle_diff_{angle_idx}_k{k}': calculate_angle_diffs(euler_angles, euler_angles[neighbors[:, k - 1]], angle_idx) for angle_idx in range(3) for k in k_values} for key, (coords, euler_angles, neighbors) in zip(dataframes.keys(), [(df[['orig_x', 'orig_y', 'orig_z']].values * pixel_size, df[['phi', 'psi', 'the']].values, cKDTree(df[['orig_x', 'orig_y', 'orig_z']].values * pixel_size).query(df[['orig_x', 'orig_y', 'orig_z']].values * pixel_size, k=max(k_values))[1]) for df in dataframes.values()])}
distances_dict = {key: {f'distance_k{k}': distances[:, k-1] for k in k_values} for key, distances in zip(dataframes.keys(), [cKDTree(df[['orig_x', 'orig_y', 'orig_z']].values * pixel_size).query(df[['orig_x', 'orig_y', 'orig_z']].values * pixel_size, k=max(k_values))[0] for df in dataframes.values()])}

""""
# PLOT THE NN DISTANCES AND ANGLE DIFFERENCE

colormap_name = 'tab10'
colormap = plt.get_cmap(colormap_name)

num_colors_needed = len(k_values)
colors = [colormap(i) for i in range(num_colors_needed)]

for i, (key, df) in enumerate(dataframes.items()):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    # Plot distances and angle differences
    for ax, label in zip(axs, ["Phi", "Psi", "Theta", "Distance"]):
        if label == "Distance":
            for j, k in enumerate(k_values):
                color = colors[j]
                sns.kdeplot(distances_dict[key][f'distance_k{k}'], color=color, label=f'k={k}', ax=ax)
        else:
            angle_idx = ["Phi", "Psi", "Theta"].index(label)
            for j, k in enumerate(k_values):
                color = colors[j]
                sns.kdeplot(angles_dict[key][f'angle_diff_{angle_idx}_k{k}'], color=color, label=f'k={k}', ax=ax)

        ax.set_xlabel('Angular Difference' if label != "Distance" else 'Distance to Nearest Neighbor (Å)')
        ax.set_ylabel('Density')
        ax.set_title(f'{key} - {label} Differences')
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
""""
