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


np.set_printoptions(precision=3)

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
    norm = plt.Normalize(0,19)
    state.plot_edges(colors=cmap(norm(mgc.labels)))

# Get colors for each label
def get_label_colors(labels, state):

    # Unique labels for each internal or external edge
    unique_int_label = np.unique(labels[state.edges_int_indices])
    unique_ext_label = np.unique(labels[state.edges_ext_indices])

    # Will hold new (renumbered) labels
    new_int_labels = np.zeros(state.M_int, dtype=int)
    new_ext_labels = np.zeros(state.M_ext, dtype=int)

    for new_label, idx in enumerate(unique_int_label):
        mask = (labels[state.edges_int_indices] == idx)
        new_int_labels[mask] = new_label

    for new_label, idx in enumerate(unique_ext_label):
        mask = (labels[state.edges_ext_indices] == idx)
        new_ext_labels[mask] = new_label


    cmap1 = mpl.cm.gist_ncar
    cmap2 = mpl.cm.tab20

    colors = np.zeros((state.n_edges, 4))

    colors[state.edges_int_indices] = cmap1(new_int_labels/(unique_int_label.size-1))
    colors[state.edges_ext_indices] = cmap2(new_ext_labels/(unique_ext_label.size-1))


    return colors


p = 6
q = 6

do_weighted = False

# For SAM schematic pattern plotting
figsize = (10,10)

mpl.rcParams.update({'axes.labelsize': 45})
mpl.rcParams.update({'xtick.labelsize': 50})
mpl.rcParams.update({'ytick.labelsize': 50})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':34})

homedir = os.environ['HOME']

energies, ols_feat_vec, states = extract_from_ds('data/sam_pattern_{:02d}_{:02d}.npz'.format(p,q))
err_energies = np.load('data/sam_pattern_{:02d}_{:02d}.npz'.format(p,q))['err_energies']
weights = 1 / err_energies**2

perf_mse_m3, perf_wt_mse_m3, perf_r2_m3, err_m3, reg_m3 = fit_multi_k_fold(ols_feat_vec, energies, weights=weights, do_weighted=do_weighted)

m2_feat = np.append(ols_feat_vec[:,0][:,None], ols_feat_vec[:,1:].sum(axis=1)[:,None], axis=1)
perf_mse_m2, perf_wt_mse_m2, perf_r2_m2, err_m2, reg_m2 = fit_multi_k_fold(m2_feat, energies, weights=weights, do_weighted=do_weighted)
perf_mse_m1, perf_wt_mse_m1, perf_r2_m1, err_m1, reg_m1 = fit_multi_k_fold(ols_feat_vec[:,0].reshape(-1,1), energies, weights=weights, do_weighted=do_weighted)
ko = ols_feat_vec[:,0]
quad_m1_feat = np.vstack((ko, ko**2)).T
perf_mse_m1_quad, perf_wt_mse_m1_quad, perf_r2_m1_quad, err_m1_quad, reg_m1_quad = fit_multi_k_fold(quad_m1_feat, energies, weights=weights, do_weighted=do_weighted)


state = states[np.argwhere(ols_feat_vec[:,0] == 0).item()]

instr = 'weighted' if do_weighted else 'no_weighted'
ds = np.load('merge_data/sam_merge_coef_class_{}_{:02d}_{:02d}.npz'.format(instr,p,q))

all_mse = ds['all_mse']
all_wt_mse = ds['all_wt_mse']

all_cv_mse = ds['all_cv_mse']
all_cv_wt_mse = ds['all_cv_wt_mse']

all_cv_mse_se = ds['all_cv_mse_se']
all_cv_wt_mse_se = ds['all_cv_wt_mse_se']

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

line_widths=np.ones(state.n_edges)*10
## Plot edges as different colors, highlight external/internal edges ##
#######################################################################
#######################################################################

plt.close('all')

state.plot(size=figsize)

colors = get_label_colors(all_mgc[0].labels, state)
line_styles = np.array(['--' for i in range(state.n_edges)])
line_styles[state.edges_int_indices] = '-'

state.plot_edges(colors=colors, line_styles=line_styles, line_widths=line_widths)
plt.savefig('{}/Desktop/fig_merge_0'.format(homedir), transparent=True)

plt.close('all')
plt.plot([0,0], [0,0], 'ko', markersize=20, label='internal node')
plt.plot([0,0], [0,0], 'rX', markersize=20, label='external node')
plt.plot([0,0], [0,0], 'k-', linewidth=6, label='internal edge')
plt.plot([0,0], [0,0], 'k--', linewidth=6, label='external edge')

plt.xlim(-100, -90)
plt.legend(loc='center')
plt.axis('off')
plt.savefig('{}/Desktop/multi_color_legend_fig'.format(homedir), transparent=True)


## After half are merged ##
#######################################################################
#######################################################################

plt.close('all')

state.plot(size=figsize)

colors = get_label_colors(all_mgc[53].labels, state)

state.plot_edges(colors=colors, line_styles=line_styles, line_widths=line_widths)

plt.savefig('{}/Desktop/fig_merge_half'.format(homedir), transparent=True)


###########################
###########################


## After 10 edge groups left ##
#######################################################################
#######################################################################

plt.close('all')

state.plot(size=figsize)

colors = get_label_colors(all_mgc[-6].labels, state)

state.plot_edges(colors=colors, line_styles=line_styles, line_widths=line_widths)

plt.savefig('{}/Desktop/fig_merge_final_10'.format(homedir), transparent=True)


###########################
###########################


## All merged - M3 ##
#######################################################################
#######################################################################

plt.close('all')

state.plot(size=figsize)

labels = all_mgc[-2].labels
assert np.unique(labels).size == 2
assert np.unique(labels[state.edges_int_indices]).size == 1


colors = np.empty(state.n_edges, dtype=object)
colors[state.edges_ext_indices] = '#ffffff'
colors[state.edges_int_indices] = '#000000'
state.plot_edges(colors=colors, line_styles=line_styles, line_widths=line_widths)

plt.savefig('{}/Desktop/fig_merge_m3'.format(homedir), transparent=True)


###########################
###########################

## One edge class: M2 ##
#######################################################################
#######################################################################

plt.close('all')

state.plot(size=figsize)

labels = all_mgc[-1].labels
assert np.unique(labels).size == 1
assert np.unique(labels[state.edges_int_indices]).size == 1


colors = np.empty(state.n_edges, dtype=object)
colors[:] = '#000000'
#colors[state.edges_int_indices] = '#ff00bf'
state.plot_edges(colors=colors, line_styles=line_styles, line_widths=line_widths)

plt.savefig('{}/Desktop/fig_merge_m2'.format(homedir), transparent=True)


###########################
###########################


#########################
plt.close('all')

fig, ax1 = plt.subplots(figsize=(7,6))

ax2 = ax1.twinx()

ax1.set_xlim(-5, all_n_params.max()+2)
ax1.plot(all_n_params[:]-2, np.sqrt(all_cv_mse[:]), 'bo', markersize=12)
ax1.plot(all_n_params[-2]-2, np.sqrt(all_cv_mse[-2]), 'rD', markersize=20)
ax1.plot(all_n_params[-1]-2, np.sqrt(all_cv_mse[-1]), 'yD', markersize=20)
#ax2.plot(all_n_params[:]-2, all_aic[:], 'k-', linewidth=4)
ax2.plot(all_n_params[:]-2, np.sqrt(all_cv_wt_mse), 'gx-', markersize=12)

ax1.set_zorder(1)
ax1.patch.set_visible(False)

#ax1.set_ylim(3.8, 10)

fig.tight_layout()

plt.savefig('{}/Desktop/merge_perf'.format(homedir), transparent=True)


###########################
###########################

# BAR chart of selected

# Grab ANN perf
n_hidden_layer = 2
n_node_hidden = 12

ds = np.load('data/sam_ann_ml_trials.npz')
all_perf_tot = ds['all_perf_tot']
all_perf_cv = ds['all_perf_cv']
all_n_params = ds['all_n_params']
trial_n_hidden_layer = ds['trial_n_hidden_layer']
trial_n_node_hidden = ds['trial_n_node_hidden']


i_hidden_layer = np.digitize(n_hidden_layer, trial_n_hidden_layer) - 1
i_node_hidden = np.digitize(n_node_hidden, trial_n_node_hidden) - 1

perf_ann_cv = all_perf_cv[i_hidden_layer, i_node_hidden]
perf_ann_tot = all_perf_tot[i_hidden_layer, i_node_hidden]
n_params_ann = all_n_params[i_hidden_layer, i_node_hidden]

##### DONE WITH ANN PERF #####

# Grab CNN perf
n_hidden_layer = 2
n_node_hidden = 4
n_conv_filters = 9

ds = np.load('data/sam_cnn_ml_trials.npz')
all_perf_tot = ds['all_perf_tot']
all_perf_cv = ds['all_perf_cv']
all_n_params = ds['all_n_params']
trial_n_hidden_layer = ds['trial_n_hidden_layer']
trial_n_node_hidden = ds['trial_n_node_hidden']
trial_n_conv_filters = ds['trial_n_conv_filters']

i_conv_filters = np.digitize(n_conv_filters, trial_n_conv_filters)
i_hidden_layer = np.digitize(n_hidden_layer, trial_n_hidden_layer) - 1
i_node_hidden = np.digitize(n_node_hidden, trial_n_node_hidden) - 1

perf_cnn_cv = all_perf_cv[i_conv_filters, i_hidden_layer, i_node_hidden]
perf_cnn_tot = all_perf_tot[i_conv_filters, i_hidden_layer, i_node_hidden]
n_params_cnn = all_n_params[i_conv_filters, i_hidden_layer, i_node_hidden]

##### DONE WITH CNN PERF #####


#########################
plt.close('all')

fig, ax = plt.subplots(figsize=(16,7))

indices = np.arange(7)

# M1, M2, M3, and all edge types separate
perfs = np.sqrt([np.mean(perf_mse_m1), np.mean(perf_mse_m1_quad), perf_ann_cv, perf_cnn_cv, all_cv_mse[0], all_cv_mse[-2], all_cv_mse[-1]])
colors = ['#888888', '#888888', '#888888', '#888888', '#888888', 'r', '#888888']
ax.bar(indices, perfs, color=colors)

# Root mean squared error in f
rms_obs = np.sqrt(np.mean(err_energies**2))

xmin, xmax = ax.get_xlim()
ax.plot([xmin,xmax], [rms_obs, rms_obs], 'k--', linewidth=6)

ax.set_xlim([xmin, xmax])
ax.set_ylim([0, 7.2])

ax.set_xticks([])

fig.tight_layout()

plt.savefig('{}/Desktop/bar_merge_perf'.format(homedir), transparent=True)

plt.close('all')


