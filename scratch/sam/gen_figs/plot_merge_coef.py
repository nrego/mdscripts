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

def plot_edge_from_mgc(state, mgc, cmap=plt.cm.tab20):
    plt.close('all')

    state.plot()
    state.plot_edges(colors=cmap(mgc.labels))


mpl.rcParams.update({'axes.labelsize': 45})
mpl.rcParams.update({'xtick.labelsize': 50})
mpl.rcParams.update({'ytick.labelsize': 50})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':14})

energies, ols_feat_vec, states = extract_from_ds('data/sam_pattern_06_06.npz')
state = states[np.argwhere(ols_feat_vec[:,0] == 0).item()]

## Plot ANN AIC as fn of hyperparams

ds = np.load('data/sam_merge_coef_data.npz')
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

ax1.scatter(all_n_params-1, all_cv_mse)
ax2.plot(all_n_params-1, all_aic)
