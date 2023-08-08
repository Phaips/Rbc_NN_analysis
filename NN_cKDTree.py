import numpy as np
import matplotlib.pyplot as plt
import starfile
from scipy.spatial import cKDTree
import seaborn as sns

# pixelsize of your tomo (here bin4)
pixel_size = 7.84
# number of nearest neighbors. here now k=2 meaning nearest neighbor. range(2,4) would be nearest and second nearest (k=3) neighbor
k_values = range(2, 3) 

# load your motl (.star file)
path = "../your/path/motl.star"
tomo = starfile.read(path)

# initialize df and sort by tomo number
dataframes = {f"df{tomo_num}": df.reset_index(drop=True) for tomo_num, df in tomo.groupby('tomo_num')}

# calculate the distance between two angles
def angular_difference(angle1, angle2):
    diff = np.abs(angle1 - angle2) % 360 # take the absolute difference of the angles. Phi angles in STOPGAP are sometimes not modulo 360
    wrap = np.minimum(diff, 360 - diff) # if 10° and 350° we want it to be 20° and not 340° :)
    return wrap

# slicing the array to loop through the angles: angle_index 0 for phi, 1 for psi, 2 for theta
def calculate_angle_diffs(euler_angles, neighbor_angles, angle_index):
    return angular_difference(euler_angles[:, angle_index], neighbor_angles[:, angle_index])

# initialize dictionaries
# find nearest neighbors using k-d-tree (space partitioning, binary trees are fast!)
angles_dict = {}
distances_dict = {}

for key, df in dataframes.items():
    coords = df[['orig_x', 'orig_y', 'orig_z']].values * pixel_size
    euler_angles = df[['phi', 'psi', 'the']].values
    tree = cKDTree(coords)
    distances, neighbors = tree.query(coords, k=max(k_values))
    
    angles_dict[key] = {}
    distances_dict[key] = {}
    for k in k_values:
        distances_dict[key][f'distance_k{k}'] = distances[:, k-1] # we start counting from 0
        
        neighbor_indices = neighbors[:, k - 1]
        neighbor_angles = euler_angles[neighbor_indices]
        
        for angle_idx in range(3):
            angle_key = f'angle_diff_{angle_idx}_k{k}'
            angle_diffs = calculate_angle_diffs(euler_angles, neighbor_angles, angle_idx)
            angles_dict[key][angle_key] = angle_diffs

""""
# Nematic (orientation) order parameter (2nd Legendre polynomial)
def orientation(angles):
    nematic = np.mean((3 * np.cos(angles)**2 - 1) / 2)
    return(nematic)

for tomo_num in unique_tomo_nums:
    print('order phi'f'{tomo_num}__:' , orientation(dataframes[f'df{tomo_num}']['phi']))
    print('order psi'f'{tomo_num}__:' , orientation(dataframes[f'df{tomo_num}']['psi']))
    print('order theta'f'{tomo_num}:' , orientation(dataframes[f'df{tomo_num}']['the']))


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
