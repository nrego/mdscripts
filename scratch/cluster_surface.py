from __future__ import division, print_function

import MDAnalysis
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from IPython import embed

import sklearn.cluster
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

univ = MDAnalysis.Universe('prot_heavies_by_charge.pdb')
mask = univ.atoms.tempfactors == 0 # Surface hydrophobic atoms

surf = univ.atoms[mask]
surf_pos = surf.positions.copy()

n_tot = surf.n_atoms

nn = NearestNeighbors(n_neighbors=6)
nn.fit(surf_pos)
dist, ind = nn.kneighbors()

# The distance of each point to its 6th nearest neighbor
#   (in decreasing order)
dist = np.sort(dist[:,-1])[::-1]

plt.plot(dist)
plt.show()

for eps in np.arange(4.0, 6.5, 0.5):
    for min_samples in np.arange(5,20,1):
        labels = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(surf_pos).labels_
        
        n_noise = (labels == -1).sum()
        clust_labels = labels[labels!=-1]
        n_clust = np.unique(clust_labels).size
        hist, _ = np.histogram(clust_labels, bins=np.arange(n_clust+1))
        try:
            n_largest_patch = hist.max()
        except ValueError:
            n_largest_patch = 0

        print("eps: {} min_n: {}  n_patch: {} largest_patch: {:0.2f} noise: {:0.2f}".format(eps, min_samples, n_clust, n_largest_patch/n_tot, n_noise/n_tot))


labels = sklearn.cluster.DBSCAN(eps=4, min_samples=5).fit(surf_pos).labels_
surf.tempfactors = labels
surf.write('surf.pdb')