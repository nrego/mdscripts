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

UNDEFINED=-2
NOISE=-1

class DualClusterer(object):

    def __init__(self, eps, min_phob_n, phob_pos, max_phil_n=float('inf'), phil_pos=None):

        self.phob_pos = phob_pos
        self.phil_pos = phil_pos
        self.eps = eps
        self.min_phob_n = min_phob_n
        self.max_phil_n = max_phil_n
        # clusterer that only considers hydrophobic atoms
        self.clust = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_phob_n).fit(phob_pos)

        self._phob_nn = None
        self._phil_nn = None

        self.core_sample_indices = None

        self.labels = np.empty((self.phob_pos.shape[0]), dtype=int)
        self.labels[:] = -2


    # Get number of hydrophobic surface atoms to be clustered
    @property
    def n_atoms(self):
        return self.phob_pos.shape[0]
    
    def is_core(self, p):
        return self.phob_nn[p].size >= self.min_phob_n and self.phil_nn[p].size < self.max_phil_n

    # Run the dbscan clustering algorithm.
    # Fits core_sample_indices_, indice_
    def cluster(self):
        clust_idx = 0
        core_indices = np.array([], dtype=int)
        for p in range(self.n_atoms):

            # point already processed - must be a neighbor of core point
            if self.labels[p] != UNDEFINED:
                continue

            # Noise point
            if not self.is_core(p):
                self.labels[p] = NOISE
                continue

            # It's a core point...
            core_indices = np.append(core_indices, p)
            self.labels[p] = clust_idx
            s = self.phob_nn[p].copy()
            idx = 0
            # Now loop thru its neighbors and assign them to this cluster
            while(idx < s.size):
                q = s[idx]
                idx += 1
                # Q is already assigned to a cluster
                # Note this implies q is *Not* a core point, or else
                # It would have already picked up point p when it was
                # assigned
                if self.labels[q] != UNDEFINED and self.labels[q] != NOISE:
                    continue

                # It was undefined or noise - put it into this cluster
                self.labels[q] = clust_idx

                if not self.is_core(q):
                    continue

                # q is a core point - add its neighbors, as well
                # Note there will be duplicates (including point p, by defn)
                #   added to the array s - not a problem, since duplicates will
                #   be skipped if they have already been labeled
                core_indices = np.append(core_indices, q)
                s = np.append(s, self.phob_nn[q])

            clust_idx += 1
        
        self.core_sample_indices = core_indices
        return self

    @property
    def phob_nn(self):
        if self._phob_nn is None:
            tree = KDTree(self.phob_pos)
            self._phob_nn = tree.query_radius(self.phob_pos, self.eps)

        return self._phob_nn

    @property
    def phil_nn(self):
        if self._phil_nn is None:
            if self.phil_pos is None:
                self._phil_nn = np.array([np.array([]) for i in range(self.n_atoms)])
            else:
                tree = KDTree(self.phil_pos)
                self._phil_nn = tree.query_radius(self.phob_pos, self.eps)

        return self._phil_nn
