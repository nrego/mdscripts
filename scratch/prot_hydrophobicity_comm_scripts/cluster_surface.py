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
import sys

logging.captureWarnings(True)

name = 'prot_heavies_by_charge.pdb'
univ = MDAnalysis.Universe(name)
phob_mask = univ.atoms.tempfactors == 0 # Surface hydrophobic atoms
phil_mask = univ.atoms.tempfactors == -1 # Surface hydrophilic atoms

surf_mask = univ.atoms.tempfactors > -2 # all surface atoms

all_surf = univ.atoms[surf_mask]
all_pos = all_surf.positions

phob_surf = univ.atoms[phob_mask]
phob_pos = phob_surf.positions

phil_surf = univ.atoms[phil_mask]
phil_pos = phil_surf.positions

n_tot = phob_surf.n_atoms + phil_surf.n_atoms
if name == 'prot_by_charge.pdb':
    surfout = 'surf.pdb'
else: 
    surfout = 'surf_h.pdb'
phob_surf.write(surfout)

try: 
    arg = sys.argv[1]
except IndexError:
    arg = None
if arg == '-c':
    u2 = MDAnalysis.Universe(surfout)
    assert u2.atoms.n_atoms == phob_surf.n_atoms
    id1 = u2.select_atoms('resid 62 and name CD1')[0].index
    id2 = u2.select_atoms('resid 21 and name CD')[0].index
    id3 = u2.select_atoms('resid 53 and name CG1')[0].index

    id4 = u2.select_atoms('resid 37 and name CD')[0].index

    for eps in np.arange(4.5, 5.0, 0.5):
        for min_samples in np.arange(1,50,1):

            for diff in [-1, 0, 1, 2, 3, 4, 5, 6]:
            #for diff in [1000]:
            
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
                if labels[id1] == labels[id2] and labels[id1] == labels[id3] and labels[id1] != -1 and (n_largest_patch/n_tot) < 0.3 and (n_largest_patch/n_tot) > 0.1 and labels[98] != -1 and labels[id4] != -1:
                    print(outstr)
                    print("  same")            
                #dbscan = sklearn.cluster.DBSCAN(eps, min_samples).fit(phob_pos)
                #assert np.array_equal(dbscan.labels_, labels)

eps = 4.5
min_samples = 5
diff = 0

clust = DualClusterer(eps, min_samples, phob_pos, min_samples+diff, phil_pos).cluster()
core_indices = clust.core_sample_indices
labels = clust.labels


tree_phob = KDTree(phob_pos)
res = tree_phob.query_radius(all_pos, r=eps)
nn_phob = []
for arr in res:
    nn_phob.append(arr.size)
nn_phob = np.array(nn_phob)

tree_phil = KDTree(phil_pos)
res = tree_phil.query_radius(all_pos, r=eps)
nn_phil = []
for arr in res:
    nn_phil.append(arr.size)
nn_phil = np.array(nn_phil)

bb = np.arange(0, max(nn_phob.max(), nn_phil.max()))
hist, bb = np.histogram(nn_phob, bins=bb)
plt.plot(bb[:-1], hist, label='phob')
hist, bb = np.histogram(nn_phil, bins=bb)
plt.plot(bb[:-1], hist, label='phil')
plt.legend()
plt.show()


cores = phob_surf[core_indices]
for core in cores:
    if core.name[0] != 'H':
        core.name = 'COR'
    else:
        core.name = 'HOR'


n_cores = core_indices.size

n_noise = (labels == -1).sum()
n_patch = (labels != -1).sum()
clust_labels = labels[labels!=-1]
n_clust = np.unique(clust_labels).size

hist, _x = np.histogram(clust_labels, bins=np.arange(n_clust+1))
try:
    n_largest_patch = hist.max()
    largest_patch_label = _x[np.argmax(hist)]
except ValueError:
    n_largest_patch = 0

phob_surf.tempfactors = labels
if name == 'prot_by_charge.pdb':
    clustout = 'cluster.pdb'
else:
    clustout = 'cluster_h.pdb'
#phob_surf[phob_surf.tempfactors == largest_patch_label].tempfactors = -2
phob_surf.write(clustout)

outstr = "eps: {} min_n: {} diff: {} n_surf: {} n_phob_surf: {} per hydrophobic: {:0.2f} n_core: {} n_patch: {} largest_patch: {:0.2f} ({}) patch: {:0.2f}".format(eps, min_samples, diff, 
                                                                                                                     n_tot, phob_surf.n_atoms, phob_surf.n_atoms / n_tot,
                                                                                                                     n_cores, n_clust, n_largest_patch/n_tot, n_largest_patch, n_patch/n_tot)
print(outstr)

