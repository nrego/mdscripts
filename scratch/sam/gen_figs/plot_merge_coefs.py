from __future__ import division

import numpy as np
import MDAnalysis

import argparse
from IPython import embed
import os, glob

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np

from scratch.sam.util import *

prop_cycle = plt.rcParams['axes.prop_cycle']
colors_cycle = prop_cycle.by_key()['color']
n = 131
color = plt.cm.coolwarm(np.linspace(0.1,0.9,n)) # This returns RGBA; convert:
colors = []
for i in range(n):
    colors.append(mpl.colors.rgb2hex(color[i]))

color = colors
convert = lambda rgb:'#%02x%02x%02x' % (rgb[0]*255,rgb[1]*255,rgb[2]*255)

from sklearn import datasets, linear_model

plt.close('all')
homedir = os.environ['HOME']
from IPython import embed
mpl.rcParams.update({'axes.labelsize': 45})
mpl.rcParams.update({'xtick.labelsize': 35})
mpl.rcParams.update({'ytick.labelsize': 35})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':10})


class Edge:
    def __init__(self, idx):
        self.idx = idx
        self.indices = [idx]
        self.parent = self

    def merge(self, other_edge):
       self.indices = np.append(self.indices, other_edge.indices)

def get_edge_from_state(state, edges):

    edge_feat_vec = np.zeros(edges.shape[0])

    for idx, (i,j) in enumerate(edges):

        assert i in state.patch_indices

        local_i = np.where(state.patch_indices == i)[0].item()

        # Pos i is a hydroxyl
        if not state.methyl_mask[local_i]:
            # If j is not in patch, we're done
            if j not in state.patch_indices:
                edge_feat_vec[idx] = 1
            else:
                local_j = np.where(state.patch_indices == j)[0].item()
                if not state.methyl_mask[local_j]:
                    edge_feat_vec[idx] = 1

    return edge_feat_vec



energies, feat_vec, states = extract_from_ds("sam_pattern_06_06.npz")

temp_state = states[-1]

edges, ext_indices = enumerate_edges(temp_state.positions, temp_state.pos_ext, temp_state.nn_ext, temp_state.patch_indices)

big_feat = np.zeros((len(states), edges.shape[0]+1))

for i, state in enumerate(states):
    big_feat[i, 0] = state.k_o
    big_feat[i, 1:] = get_edge_from_state(state, edges)

    assert state.n_oo + state.n_oe == big_feat[i,1:].sum()


myfeat = big_feat.copy()
reg = linear_model.Ridge(fit_intercept=True)

errs = np.zeros(edges.shape[0])
aics = np.zeros_like(errs)

perf_mse_tot = np.zeros_like(aics)

edge_list = np.empty(edges.shape[0], dtype=object)
for i in range(edges.shape[0]):
    edge_list[i] = Edge(i)


for i_round in range(edges.shape[0]-1):
    print("round {}".format(i_round))
    reg.fit(myfeat, energies)

    pred = reg.predict(myfeat)
    err = ((energies - pred)**2).mean()
    errs[i_round] = err
    coef = reg.coef_[1:]

    aics[i_round] = aic_ols(reg, err)

    min_dist = np.inf
    min_i = None
    min_j = None

    perf_mse, err, xvals, fit, reg = fit_leave_one(myfeat, energies)
    perf_mse_tot[i] = perf_mse.mean()

    for i in range(coef.shape[0]):
        for j in range(i+1, coef.shape[0]):

            dist = (coef[i] - coef[j])**2

            if dist < min_dist:
                min_dist = dist
                min_i = i
                min_j = j
    
    edge_i = edge_list[min_i]
    edge_j = edge_list[min_j]

    edge_i.merge(edge_j)
    edge_list = np.delete(edge_list, [min_i, min_j])

    edge_list = np.append(edge_list, edge_i)

    new_feat = np.zeros((myfeat.shape[0], myfeat.shape[1]-1))
    new_feat[:,0] = myfeat[:,0]
    new_feat[:,1:-1] = np.delete(myfeat, [0,min_i+1,min_j+1], axis=1)
    new_feat[:,-1] = myfeat[:,min_i+1] + myfeat[:,min_j+1]

    myfeat = new_feat

    edge_indices = [edge.indices for edge in edge_list]

    colors = np.empty(edges.shape[0], dtype=object)
    styles = np.empty(edges.shape[0], dtype=object)

    for i, edge in enumerate(edge_list):
        if len(edge.indices) == 1:
            colors[edge.indices] = 'k'
            styles[edge.indices] = '--'
        else:
            idx = i % 10
            styles[edge.indices] = '-'
            colors[edge.indices] = colors_cycle[idx]

    plt.close('all')

    plot_edge_list(states[0].pos_ext, edges, states[0].patch_indices, do_annotate=False, colors=colors, line_styles=styles)

    plt.savefig('{}/Desktop/fig_{}'.format(os.environ['HOME'], i_round))

reg.fit(myfeat, energies)
pred = reg.predict(myfeat)
err = ((energies - pred)**2).mean()
errs[i_round+1] = err
coef = reg.coef_
aics[i_round+1] = aic_ols(reg, err)


