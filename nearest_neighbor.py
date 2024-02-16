import numpy as np
import matplotlib.pyplot as plt
import starfile
from scipy.spatial import cKDTree

"""
Give path to your starfile motive/particle list
!! Depending on the format (RELION, WARP/M, STOPGAP, PYTOM etc.) adjust the "tomo_num" and "angle" names !!
"""
path = "/.../all_motl/particles.star"
tomo = starfile.read(path)

dataframes = {tomo_num: df.reset_index(drop=True) 
              for tomo_num, df in tomo.groupby('ptmMicrographName')} # Adjust accordingly to your motive list (i.e. rlnMicrographName or tomo_num)

# Pixel/Voxel size at your tomogram binning
pixel_size = 7.84
k_values = range(2, 6) # This will plot the first 4 neighbors (nearest, second nearest, etc.). Change to the desired number (always start at 2!)

colormap = plt.get_cmap('tab10')
colors = [colormap(i) for i in range(len(k_values))]

distances_dict = {}
for key, df in dataframes.items():
    coords = df[['ptmCoordinateX', 'ptmCoordinateY', 'ptmCoordinateZ']].values * pixel_size # Adjust accordingly (i.e. rlnCoordinateX, orig_x, etc.) !!!
    tree = cKDTree(coords) # Binary trees are fast! :)
    distances, _ = tree.query(coords, k=max(k_values))
    distances_dict[key] = {f'distance_k{k}': distances[:, k-1] for k in k_values}


n_tomograms = len(dataframes)
# One distance and three angle plots
n_columns = 4
n_rows = n_tomograms  # One row per tomogram

fig, axes = plt.subplots(n_rows, n_columns, figsize=(20, 4 * n_rows))
if n_tomograms == 1:
    axes = np.array([axes])

for idx, key in enumerate(distances_dict):
    ax_distance = axes[idx, 0] if n_tomograms > 1 else axes[0]
    
    for k, color in zip(k_values, colors):
        distances_k = distances_dict[key][f'distance_k{k}']
        ax_distance.hist(distances_k, 
                         bins=100, 
                         color=color, 
                         alpha=0.5, 
                         edgecolor='black',
                         linewidth=0.5,
                         label=f'k={k}: {np.mean(distances_k):.2f}Å ± {np.std(distances_k):.2f}Å')
        
    ax_distance.set_xlabel('Distance (Å)')
    ax_distance.set_ylabel('Count')
    ax_distance.set_title(f'Distances for tomo {key}')
    ax_distance.set_xlim(100, 400) # Adjust based on your data range
    ax_distance.legend()
    
    angle_columns = ['ptmAngleRot', 'ptmAngleTilt', 'ptmAnglePsi'] # Change angle name accordingly (i.e. rlnAngleRot, orig_x) !!!
    angle_names = ['Rot', 'Tilt', 'Psi'] # This could be different depending on the Euler angle convention !!!
    for angle_idx, angle_col in enumerate(angle_columns):
        ax_angle = axes[idx, angle_idx + 1] if n_tomograms > 1 else axes[angle_idx + 1]
        dataframes[key][angle_col].hist(ax=ax_angle, 
                                        bins=30, 
                                        color=colors[angle_idx], 
                                        alpha=0.7)
        ax_angle.set_title(f'{angle_names[angle_idx]} Angle for tomo {key}')
        ax_angle.set_xlabel('Angle (degrees)')
        ax_angle.set_ylabel('Count')

plt.tight_layout()
plt.show()
