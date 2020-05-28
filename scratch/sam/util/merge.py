from __future__ import division

import numpy as np
import MDAnalysis

import argparse
from IPython import embed
import os, glob

import time

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

from scratch.sam.util import *

from scratch.neural_net.lib import *
import copy

## Encapsulate merging of edge types, or any arbitrary feature
# Label is arbitrary; can only merge groups that have the same label
class MergeGroup:

    def __init__(self, indices, label='A'):
        
        self.label = label

        self.indices = np.array(indices, ndmin=1)
        self.indices.sort()
        if not np.array_equal(self.indices, np.unique(self.indices)):
            raise ValueError('Repeated indices')

    @classmethod
    def merge_groups(cls, grp1, grp2):
        assert isinstance(grp1, cls) and isinstance(grp2, cls)

        if grp1.label != grp2.label:
            raise ValueError('Different group labels')

        return cls(np.append(grp1.indices, grp2.indices), label=grp1.label)

    def __len__(self):
        return self.indices.size

    def __repr__(self):
        return "<MergeGroup; Class: {}  Indices: {}>".format(self.label, self.indices)

class MergeGroupCollection:

    def __init__(self):
        self.groups = []

    def __len__(self):
        return len(self.groups)
    
    def add_group(self, grp):
        self.groups.append(grp)

    def add_from_labels(self, labels):
        edge_indices = np.arange(labels.size)

        for i_label in np.unique(labels):
            mask = labels == i_label
            self.add_group(MergeGroup(edge_indices[mask]))

    ## Attempt to merge two groups; does nothing if 2 groups
    #     cannot be merged (e.g., if they're the same or )
    def merge_groups(self, idx1, idx2):

        try:
            grp1 = self.groups[idx1]
            grp2 = self.groups[idx2]

            new_grp = MergeGroup.merge_groups(grp1, grp2)

            self.groups.remove(grp1)
            self.groups.remove(grp2)
            self.add_group(new_grp)

        except:
            return

    @property
    def n_indices(self):
        return np.sum([len(grp) for grp in self.groups])
    
    # Return a list of labels for all of the items in all
    #   of the merge groups, according to whichever group they belong to
    @property
    def labels(self):

        labels = np.zeros(self.n_indices, dtype=int)
        labels[:] = -1

        for i, grp in enumerate(self.groups):
            labels[grp.indices] = i

        return labels





