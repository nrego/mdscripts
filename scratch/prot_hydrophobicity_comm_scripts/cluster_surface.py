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
import logging

logging.captureWarnings(True)

name = 'prot_by_charge.pdb'
univ = MDAnalysis.Universe(name)
phob_mask = univ.atoms.tempfactors == 0 # Surface hydrophobic atoms
phil_mask = univ.atoms.tempfactors == -1 # Surface hydrophilic atoms

phob_surf = univ.atoms[phob_mask]
phob_pos = phob_surf.positions

phil_surf = univ.atoms[phil_mask]
phil_pos = phil_surf.positions

n_tot = phob_surf.n_atoms 
if name == 'prot_by_charge.pdb':
    surfout = 'surf.pdb'
else: 
    surfout = 'surf_h.pdb'
phob_surf.write(surfout)

'''
u2 = MDAnalysis.Universe(surfout)
assert u2.atoms.n_atoms == phob_surf.n_atoms
id1 = u2.select_atoms('resid 62 and name CD1')[0].index
id2 = u2.select_atoms('resid 21 and name CD')[0].index
id3 = u2.select_atoms('resid 53 and name CG1')[0].index

id4 = u2.select_atoms('resid 37 and name CD')[0].index

for eps in np.arange(4.0, 6.5, 0.5):
    for min_samples in np.arange(1,12,1):

        for diff in [-1, 0, 1, 2, 3, 4, 5, 6]:
        
            clust = DualClusterer(eps, min_samples, phob_pos, min_samples+diff, phil_pos).cluster()
            labels = clust.labels
            core_indices = clust.core_sample_indices
            n_cores = core_indices.size
            
            n_noise = (labels == -1).sum()
            clust_labels = labels[labels!=-1]
            n_clust = np.unique(clust_labels).size
            hist, _x = np.histogram(clust_labels, bins=np.arange(n_clust+1))
            try:
                n_largest_patch = hist.max()
            except ValueError:
                n_largest_patch = 0

            outstr = "eps: {} min_n: {} diff: {} n_surf: {} n_core: {} n_patch: {} largest_patch: {:0.2f} noise: {:0.2f}".format(eps, min_samples, diff, phob_surf.n_atoms, n_cores, n_clust, n_largest_patch/n_tot, n_noise/n_tot)
            #print(outstr)
            if labels[id1] == labels[id2] and labels[id1] == labels[id3] and labels[id1] != -1 and (n_largest_patch/n_tot) < 0.4 and (n_largest_patch/n_tot) > 0.1 and labels[98] != -1 and labels[id4] != -1:
                print(outstr)
                print("  same")            
            #dbscan = sklearn.cluster.DBSCAN(eps, min_samples).fit(phob_pos)
            #assert np.array_equal(dbscan.labels_, labels)
'''

eps = 4.5
min_samples = 6
diff = 0

clust = DualClusterer(eps, min_samples, phob_pos, min_samples+diff, phil_pos).cluster()
core_indices = clust.core_sample_indices
labels = clust.labels

cores = phob_surf[core_indices]
for core in cores:
    if core.name[0] != 'H':
        core.name = 'COR'
    else:
        core.name = 'HOR'

phob_surf.tempfactors = labels
if name == 'prot_by_charge.pdb':
    clustout = 'cluster.pdb'
else:
    clustout = 'cluster_h.pdb'
phob_surf.write(clustout)

n_cores = core_indices.size

n_noise = (labels == -1).sum()
clust_labels = labels[labels!=-1]
n_clust = np.unique(clust_labels).size
hist, _x = np.histogram(clust_labels, bins=np.arange(n_clust+1))
try:
    n_largest_patch = hist.max()
except ValueError:
    n_largest_patch = 0

outstr = "eps: {} min_n: {} diff: {} n_surf: {} n_core: {} n_patch: {} largest_patch: {:0.2f} noise: {:0.2f}".format(eps, min_samples, diff, phob_surf.n_atoms, n_cores, n_clust, n_largest_patch/n_tot, n_noise/n_tot)
print(outstr)

