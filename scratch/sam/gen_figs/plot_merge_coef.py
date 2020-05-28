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

import itertools

from sklearn import datasets, linear_model
from sklearn.cluster import AgglomerativeClustering

def aic(n_samples, mse, k_params, do_corr=False):

    if do_corr:
        corr = (2*k_params**2 + 2*k_params) / (n_samples-k_params-1)
    else:
        corr = 0

    return n_samples*np.log(mse) + 2*k_params + corr

# Construct a reduced feature set by summing over features
#    with like labels
def construct_red_feat(feat_vec, labels):

    n_sample = feat_vec.shape[0]
    # Number of unique features after merging based on labels
    n_clust = np.unique(labels).size

    red_feat = np.zeros((n_sample, n_clust))

    for i_clust in range(n_clust):
        mask = (labels == i_clust)
        red_feat[:, i_clust] = feat_vec[:,mask].sum(axis=1)

    return red_feat

def plot_mgc(state, mgc, cmap=plt.cm.tab20):
    plt.close('all')

    state.plot()
    state.plot_edges(colors=cmap(mgc.labels))

def get_reg(feat_vec, energies, mgc):
    red_feat = construct_red_feat(feat_vec, np.append(mgc.labels, mgc.labels.max()+1))

    reg = linear_model.LinearRegression()
    reg.fit(red_feat, energies)

    pred = reg.predict(red_feat)

    err = energies - pred

    return reg, err


p = 4
q = 9

mpl.rcParams.update({'axes.labelsize': 45})
mpl.rcParams.update({'xtick.labelsize': 50})
mpl.rcParams.update({'ytick.labelsize': 50})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':14})

energies, ols_feat_vec, states = extract_from_ds('data/sam_pattern_{:02d}_{:02d}.npz'.format(p,q))
err_energies = np.load('data/sam_pattern_{:02d}_{:02d}.npz'.format(p,q))['err_energies']



state = states[np.argwhere(ols_feat_vec[:,0] == 0).item()]

## Plot ANN AIC as fn of hyperparams

ds = np.load('merge_data/sam_merge_coef_{:02d}_{:02d}.npz'.format(p,q))
all_mse = ds['all_mse']
all_cv_mse = ds['all_cv_mse']
# Number of edge classes + 2 (for ko and the intercept)
all_n_params = ds['all_n_params']
all_mgc = ds['all_mgc']

# Contains h_k_oo for each edge, plus ko
#    shape: (M_tot + 1)
full_feat_vec = ds['feat_vec']


n_dat = full_feat_vec.shape[0]

all_aic = aic(n_dat, all_mse, all_n_params, do_corr=True)

homedir = os.environ['HOME']

cmap = mpl.cm.tab20
norm = plt.Normalize(0,19)



plt.close('all')

fig, ax1 = plt.subplots(figsize=(7,6))

ax2 = ax1.twinx()

ax1.plot([0, all_n_params.max()+2], [np.mean(err_energies**2),np.mean(err_energies**2)], '--')
ax1.set_xlim(0, all_n_params.max()+2)
ax1.scatter(all_n_params-1, all_cv_mse)
ax2.plot(all_n_params-1, all_aic)



tmp_labels = np.zeros_like(all_mgc[0].labels)
tmp_labels[state.edges_int_indices] = 1
tmp_labels[state.edges_periph_periph_indices] = 2

red_feat = construct_red_feat(full_feat_vec, np.append(tmp_labels, tmp_labels.max()+1))

perf, err, _, _, reg = fit_k_fold(red_feat, energies)


