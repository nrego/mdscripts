from __future__ import division, print_function

import MDAnalysis
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from IPython import embed

import sklearn.cluster
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph


univ = MDAnalysis.Universe('prot_heavies_by_charge.pdb')
mask = univ.atoms.tempfactors == 0 # Surface hydrophobic atoms

surf = univ.atoms[mask]
surf_pos = surf.positions.copy()

bandwidth = sklearn.cluster.estimate_bandwidth(surf_pos, quantile=0.3)

connectivity = kneighbors_graph(surf_pos, n_neighbors=10, include_self=False)

scores = []



for eps in np.arange(0.5, 5.5, 0.5):
    for min_samples in np.arange(5, 13, 1):
        labels = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(surf_pos).labels_
        print("eps: {} n: {}  k: {}".format(eps, min_samples, np.unique(labels).size))

embed()
'''
kvals = range(2,100)
for k in kvals:
    clust = sklearn.cluster.AgglomerativeClustering(n_clusters=k, linkage='ward', connectivity=connectivity)
    clust.fit(surf_pos)
    labels = clust.labels_
    s = metrics.silhouette_score(surf_pos, labels, metric='euclidean')
    print("k: {}, s: {}".format(k,s))
    scores.append(s)

scores = np.array(scores)

max_idx = np.argmax(scores)
print('best performance: k={} clusters'.format(kvals[max_idx]))
'''


