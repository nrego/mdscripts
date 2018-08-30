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


class DualClusterer(object):

    def __init__(self, eps, min_samples, phob_pos, phil_pos=None):

        self._phob_pos = phob_pos
        self._phil_pos = phil_pos
        self._eps = eps
        self._min_samples = min_samples

        # clusterer that only considers hydrophobic atoms
        self.clust = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(phob_pos)

        self.tree = KDTree(phob_pos)
        self._phob_nn = None
        self._phil_nn = None

        self._core_sample_indices = None


    def _get_core_sample_indices(self):
        phob_nl = self.phob_nn
        phil_nl = self.phil_nn

        if phil_nl is None:
            self._core_sample_indices = self.clust.core_sample_indices_

        else:
            core_indices = []
            for idx in self.clust.core_sample_indices_:
                n_phob = phob_nl[idx].size
                n_phil = phil_nl[idx].size

                if n_phob - n_phil >= self.min_samples:
                    core_indices.append(idx)

            self._core_sample_indices = np.array(core_indices).astype(int)

    @property
    def labels_(self):
        return self.clust.labels_

    @property
    def core_sample_indices_(self):
        if self._core_sample_indices is None:
            self._get_core_sample_indices()

        return self._core_sample_indices

    @property
    def phob_pos(self):
        return self._phob_pos

    @property
    def phil_pos(self):
        return self._phil_pos

    @property
    def eps(self):
        return self._eps

    @property
    def min_samples(self):
        return self._min_samples

    @property
    def phob_nn(self):
        if self._phob_nn is None:
            self._phob_nn = self.tree.query_radius(self.phob_pos, self.eps)
        return self._phob_nn

    @property
    def phil_nn(self):
        if self._phil_nn is None and self.phil_pos is not None:
            self._phil_nn = self.tree.query_radius(self.phil_pos, self.eps)
        return self._phil_nn

    
    
    
    

