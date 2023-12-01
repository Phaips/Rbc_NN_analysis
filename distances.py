import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
import starfile

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x-cen)**2 / (2*wid**2))

pixel_size = 3.92
k_values = range(2, 6)
colormap = plt.get_cmap('tab10')
colors = [colormap(i) for i in range(len(k_values))]

path = "path/to/motl.star"
tomo = starfile.read(path)
dataframes = {tomo_num: df.reset_index(drop=True) for tomo_num, df in tomo.groupby('tomo_num')} # STOPGAP format can change to _ptm or _rln Micrographname etc.

distances_dict = {}
for key, df in dataframes.items():
    coords = df[['orig_x', 'orig_y', 'orig_z']].values * pixel_size # STOPGAP format can change to CoordinateX,Y etc.
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=max(k_values))
    distances_dict[key] = {f'distance_k{k}': distances[:, k-1] for k in k_values}

n_tomograms = len(dataframes)
n_columns = 2 
n_rows = (n_tomograms + 1) // n_columns if n_tomograms % n_columns else n_tomograms // n_columns

fig, axes = plt.subplots(n_rows, n_columns, figsize=(15, 5 * n_rows))
axes = axes.flatten() if n_tomograms > 1 else [axes]

for idx, (key, ax_distance) in enumerate(zip(distances_dict, axes)):
    labels = []
    for k, color in zip(k_values, colors):
        distances_k = distances_dict[key][f'distance_k{k}']
        n, bins, _ = ax_distance.hist(distances_k, bins=100, color=color, alpha=0.5, edgecolor='black', linewidth=0.5)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        p0 = [n.max(), bin_centers[n.argmax()], distances_k.std()]
        popt, _ = curve_fit(gaussian, bin_centers, n, p0=p0)
        x_fit = np.linspace(bin_centers.min(), bin_centers.max(), 1000)
        y_fit = gaussian(x_fit, *popt)
        ax_distance.plot(x_fit, y_fit, color=color, lw=2)
        label = f'k={k}: {popt[1]:.2f}Å ± {np.abs(popt[2]):.2f}Å'
        labels.append(label)
    ax_distance.set_xlabel('Distance (Å)')
    ax_distance.set_ylabel('Count')
    ax_distance.set_title(f'Distances for tomo {key}')
    ax_distance.set_xlim(100, 300)
    ax_distance.legend(labels, loc='upper right')

plt.tight_layout()
plt.show()
