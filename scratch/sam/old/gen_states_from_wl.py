from __future__ import division, print_function

import numpy as np
from wang_landau import WangLandau
from scipy.special import binom
from itertools import combinations

import networkx as nx
from scipy.spatial import cKDTree

from matplotlib import pyplot as plt

import matplotlib as mpl

from IPython import embed

import cPickle as pickle

import os

mpl.rcParams.update({'axes.labelsize': 60})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 40})
mpl.rcParams.update({'axes.titlesize': 50})
mpl.rcParams.update({'legend.fontsize':20})

## Driver script to generate SAM patterns (configurations) ##
##   that span some order parameter using Wang-Landau ##

positions = np.load('../analysis_data.dat.npz')['positions']

## d_ch3 ##
def get_methyl_rms(positions, methyl_mask, prec=2):
    pos = positions[methyl_mask]

    if pos.size == 0:
        return 0.0
    centroid = pos.mean(axis=0)
    diff = pos-centroid
    sq_dev = (diff**2).sum(axis=1)

    return np.round(np.sqrt(sq_dev.mean()), prec)

## k_eff ##
def get_keff(positions, methyl_mask):
    indices_all = np.arange(36)
    indices_ch3 = methyl_mask
    indices_oh = np.delete(indices_all, methyl_mask)

    tree = cKDTree(positions)
    edges = list(tree.query_pairs(r=0.6))

    graph = nx.Graph()
    #node_dict = [(idx, dict(phob=(idx in indices_ch3))) for idx in indices_all]
    graph.add_nodes_from(indices_all)
    
    for i,j in edges:
        if i in indices_ch3 and j in indices_ch3:
            weight = 1
        else:
            weight = 0

        graph.add_edge(i,j,weight=weight)

    edgewt = np.array(dict(graph.degree(weight='weight')).values()).sum()

    return edgewt

def plot_pattern(positions, methyl_mask, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))

    ax.set_xticks([])
    ax.set_yticks([])
    pos = positions[methyl_mask]
    ax.plot(positions[:,0], positions[:,1], 'bo', markersize=18)
    ax.plot(pos[:,0], pos[:,1], 'ko', markersize=18)
    plt.show()

max_val = 1.75
rms_bins = np.arange(0, max_val+0.1, 0.05)

max_keff = get_keff(positions, np.arange(36))
k_bins = np.arange(0,max_keff+2,2)

### TESTING WL ###
#wl = WangLandau(positions, get_methyl_rms, rms_bins, eps=1e-6)
#wl.gen_states(k=4, do_brute=True)
#d = wl.density.copy()
#wl.gen_states(k=4, do_brute=False)

wl = WangLandau(positions, get_keff, k_bins, eps=1e-6, f_init=8, f_scale=0.5)
wl.gen_states(k=18, do_brute=False)
