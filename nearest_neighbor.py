import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np
import matplotlib.pyplot as plt
import starfile
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
ordinal = lambda n: f"{n}{'th' if 11<=n%100<=13 else {1:'st',2:'nd',3:'rd'}.get(n%10,'th')}"

star_path = "/scicore/home/engel0006/vysamu91/aretomo3/chlorella/rln5/tm/out/particles.star"
k_max = 6
xlim = (50, 250)

df = starfile.read(star_path)
coords = {t: g[["rlnCenteredCoordinateXAngst",
                "rlnCenteredCoordinateYAngst",
                "rlnCenteredCoordinateZAngst"]].values
          for t, g in df.groupby("rlnTomoName")}
dist = {t: cKDTree(c).query(c, k_max+1)[0] for t, c in coords.items()}

neighbors = range(2, k_max+1)
cmap = plt.get_cmap('tab10')
colors = [cmap(i) for i in range(len(neighbors))]

for tomo, dmat in dist.items():
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    labels0, labels1 = [], []
    x = np.linspace(xlim[0], xlim[1], 200)
    for j, col in zip(neighbors, colors):
        vals = dmat[:, j]
        n, bins = np.histogram(vals, bins=100)
        centers = 0.5*(bins[1:]+bins[:-1])
        peak0 = centers[n.argmax()]
        kde = gaussian_kde(vals)
        y = kde(x)
        peak1 = x[y.argmax()]
        std = vals.std()
        ax[0].hist(vals, bins=100, alpha=0.5, edgecolor='k', color=col)
        ax[1].hist(vals, bins=100, density=True, alpha=0.3, edgecolor='k', color=col)
        ax[1].plot(x, y, color=col)
        labels0.append(f"{ordinal(j-1)} NN: {peak0:.1f}±{std:.1f}Å")
        labels1.append(f"{ordinal(j-1)} NN: {peak1:.1f}±{std:.1f}Å")
    ax[0].set(title=tomo, xlabel="Distance (Å)", ylabel="Count", xlim=xlim)
    ax[1].set(title=tomo, xlabel="Distance (Å)", ylabel="Density", xlim=xlim)
    ax[0].legend(labels0, loc='upper right')
    ax[1].legend(labels1, loc='upper right')
plt.show()
