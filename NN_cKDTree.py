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
    diff = np.abs(angle1 - angle2) % 360
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
mean_distances = {}

for key, df in dataframes.items():
    coords = df[['orig_x', 'orig_y', 'orig_z']].values * pixel_size
    euler_angles = df[['phi', 'psi', 'the']].values
    # spate partitioning to find nearest neighbors
    tree = cKDTree(coords)
    distances, neighbors = tree.query(coords, k=max(k_values))
    
    angles_dict[key] = {}
    distances_dict[key] = {}
    for k in k_values:
        # find angle of nearest neighbors
        neighbor_indices = neighbors[:, k - 1]
        neighbor_angles = euler_angles[neighbor_indices]
        #calculate their difference
        angle_diffs = calculate_angle_diffs(euler_angles, neighbor_angles, angle_index)
        angles_dict[key][f'angle_diff_{angle_index}_k{k}'] = angle_diffs
        distances_dict[key][f'distance_k{k}'] = distances[:, k-1]



"""
PLOT THE NN DISTANCES AND ANGLE DIFFERENCE

for i, (key, df) in enumerate(dataframes.items()):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Plot the histogram for distances
    ax_distances = axs[0]
    for j, k in enumerate(k_values):
        color = colors[j]
        ax_distances.hist(
            distances_dict[key][f'distance_k{k}'], 
            bins=50, color=color, alpha=0.5, label=f'k={k}'
        )
    
    ax_distances.set_xlabel('Distance to Nearest Neighbor (Å)')
    ax_distances.set_ylabel('Particle number')
    ax_distances.set_title(f'{key}')
    ax_distances.legend(loc='upper right')

    # Plot the histogram for selected angle
    ax_angles = axs[1]
    for j, k in enumerate(k_values):
        color = colors[j]
        ax_angles.hist(
            angles_dict[key][f'angle_diff_{angle_index}_k{k}'], 
            bins=50, color=color, alpha=0.5, label=f'k={k}'
        )
    
    ax_angles.set_xlabel('Angular Difference')
    ax_angles.set_ylabel('Frequency')
    ax_angles.set_title(f'{key}')
    ax_angles.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

"""


