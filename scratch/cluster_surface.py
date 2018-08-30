from __future__ import division, print_function

import MDAnalysis
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from IPython import embed

import sklearn.cluster
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, KDTree

from clustclass import DualClusterer

univ = MDAnalysis.Universe('prot_heavies_by_charge.pdb')
phob_mask = univ.atoms.tempfactors == 0 # Surface hydrophobic atoms
phil_mask = univ.atoms.tempfactors == -1 # Surface hydrophilic atoms

phob_surf = univ.atoms[phob_mask]
phob_pos = phob_surf.positions

phil_surf = univ.atoms[phil_mask]
phil_pos = phil_surf.positions

n_tot = phob_surf.n_atoms


for eps in [4.5]:
    for min_samples in np.arange(6,7,1):
        clust = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(phob_pos)

        labels = clust.labels_
        core_indices = clust.core_sample_indices_
        n_cores = core_indices.size
        
        n_noise = (labels == -1).sum()
        clust_labels = labels[labels!=-1]
        n_clust = np.unique(clust_labels).size
        hist, _ = np.histogram(clust_labels, bins=np.arange(n_clust+1))
        try:
            n_largest_patch = hist.max()
        except ValueError:
            n_largest_patch = 0

        print("eps: {} min_n: {}  n_core: {} n_patch: {} largest_patch: {:0.2f} noise: {:0.2f}".format(eps, min_samples, n_cores, n_clust, n_largest_patch/n_tot, n_noise/n_tot))

eps = 4.5
min_samples = 7


clust = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(phob_pos)
labels = clust.labels_
core_indices = clust.core_sample_indices_

clust = DualClusterer(eps, min_samples, phob_pos, phil_pos)

cores = phob_surf[core_indices]
cores.names = 'COR'
phob_surf.tempfactors = labels
embed()
phob_surf.write('cluster.pdb')

