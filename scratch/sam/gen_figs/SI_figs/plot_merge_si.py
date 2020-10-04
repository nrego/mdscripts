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

def get_reg(feat_vec, energies, mgc, k=5):
    red_feat = construct_red_feat(feat_vec, np.append(mgc.labels, mgc.labels.max()+1))

    all_perf = np.zeros(100)
    for i in range(100):
        perf, err, _, _, reg = fit_k_fold(red_feat, energies, k=k)
        all_perf[i] = perf.mean()

    return all_perf, err, reg

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

perf_m3, err_m3, _, _, reg_m3 = fit_k_fold(ols_feat_vec, energies)

state = states[np.argwhere(ols_feat_vec[:,0] == 0).item()]


ds = np.load('merge_data/sam_merge_coef_class_{:02d}_{:02d}.npz'.format(p,q))
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


## Plot edges as different colors, highlight external/internal edges ##
#######################################################################
#######################################################################

plt.close('all')

state.plot(size=figsize)

colors = get_label_colors(all_mgc[0].labels, state)
line_styles = np.array(['--' for i in range(state.n_edges)])
line_styles[state.edges_int_indices] = '-'

state.plot_edges(colors=colors, line_styles=line_styles, line_widths=np.ones(state.n_edges)*6)
plt.savefig('/Users/nickrego/Desktop/fig_merge_0', transparent=True)

plt.close('all')
plt.plot([0,0], [0,0], 'ko', markersize=20, label='internal node')
plt.plot([0,0], [0,0], 'rX', markersize=20, label='external node')
plt.plot([0,0], [0,0], 'k-', linewidth=6, label='internal edge')
plt.plot([0,0], [0,0], 'k--', linewidth=6, label='external edge')

plt.xlim(-100, -90)
plt.legend(loc='center')
plt.axis('off')
plt.savefig('/Users/nickrego/Desktop/multi_color_legend_fig', transparent=True)




## After merger number 1 ##
#######################################################################
#######################################################################

plt.close('all')

state.plot(size=figsize)

colors = get_label_colors(all_mgc[1].labels, state)

fresh_merge_mask = all_mgc[1].labels == all_mgc[1].labels.max()

line_widths=np.ones(state.n_edges)*6
line_widths[fresh_merge_mask] = 12
state.plot_edges(colors=colors, line_styles=line_styles, line_widths=line_widths)

plt.savefig('/Users/nickrego/Desktop/fig_merge_1', transparent=True)

##################################################
##################################################



## After half are merged ##
#######################################################################
#######################################################################

plt.close('all')

state.plot(size=figsize)

colors = get_label_colors(all_mgc[53].labels, state)

line_widths=np.ones(state.n_edges)*6

state.plot_edges(colors=colors, line_styles=line_styles, line_widths=line_widths)

plt.savefig('/Users/nickrego/Desktop/fig_merge_half', transparent=True)


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

line_widths=np.ones(state.n_edges)*6

colors = np.empty(state.n_edges, dtype=object)
colors[state.edges_ext_indices] = '#ffa200'
colors[state.edges_int_indices] = '#ff00bf'
state.plot_edges(colors=colors, line_styles=line_styles, line_widths=line_widths)

plt.savefig('/Users/nickrego/Desktop/fig_merge_m3', transparent=True)


###########################
###########################


#########################
plt.close('all')

fig, ax1 = plt.subplots(figsize=(7,6))

ax2 = ax1.twinx()

ax1.set_xlim(-5, all_n_params.max()+2)
ax1.plot(all_n_params[:-1]-2, all_cv_mse[:-1], 'bo')
ax1.plot(all_n_params[-2]-2, all_cv_mse[-2], 'rD', markersize=20)
ax2.plot(all_n_params[:-1]-2, all_aic[:-1], 'k-', linewidth=4)

ax1.set_zorder(1)
ax1.patch.set_visible(False)


fig.tight_layout()

plt.savefig('/Users/nickrego/Desktop/merge_perf', transparent=True)

#### DO MERGING BY EDGE CLASSES: Automerge edges based on class:
# periph-external edges
# periph-periph edges
# periph-buried edges
# buried-buried edges

## Exhaustively split by edge type ##

this_labels = np.zeros_like(all_mgc[0].labels)

this_labels[state.edges_ext_indices] = 0
this_labels[state.edges_periph_periph_indices] = 1
this_labels[state.edges_periph_buried_indices] = 2
this_labels[state.edges_buried_buried_indices] = 3

edge_mgc = MergeGroupCollection()
edge_mgc.add_group(MergeGroup(state.edges_ext_indices))
edge_mgc.add_group(MergeGroup(state.edges_periph_periph_indices))
edge_mgc.add_group(MergeGroup(state.edges_periph_buried_indices))
edge_mgc.add_group(MergeGroup(state.edges_buried_buried_indices))

#edge_mgc.add_from_labels(this_labels)

labels = {
    0: 'ext',
    1: 'periph_periph',
    2: 'periph_buried',
    3: 'buried_buried'
}

## 4 categories ##
this_mgc = copy.deepcopy(edge_mgc)

perf_4, err_4, reg_4 = get_reg(full_feat_vec, energies, edge_mgc)

print("\n4 edge classes ")
print("##############")
print(" perf: {:.2f}  cv: {:.2f}  ".format(np.mean(err_4**2), np.mean(perf_4)))

## 3 categories ##
## this is 4 choose 2 - the 2 go in one cat,
#. the remaining 2 go into the remaining 2 cats


print("\n3 edge classes ")
print("##############")

dict_3 = dict()
cv_dict_3 = dict()

for (i,j) in itertools.combinations(labels.keys(), 2):
    
    this_mgc = copy.deepcopy(edge_mgc)
    this_mgc.merge_groups(i,j)
    perf, err, reg = get_reg(full_feat_vec, energies, this_mgc)

    dict_3[(i,j)] = np.mean(err**2)
    cv_dict_3[(i,j)] = np.mean(perf)
    print("\n merged [{} and {}]".format(labels[i], labels[j]))
    print(" perf: {:.2f}  cv: {:.2f}  reg: {}".format(np.mean(err**2), np.mean(perf), reg.coef_))


## 2 categories ##
## this is 0.5*(4 choose 2) - the 2 go in one cat,
#. the remaining 2 go into the remaining cat. 
#.  factor of '0.5' because order of category assignment 
#.  doesn't matter.
#
#  *Plus* (4 choose 1) - 1 goes in one cat, the other 3 in the other

print("\n2 edge classes ")
print("##############")

dict_2 = dict()
cv_dict_2 = dict()

for (i,j) in itertools.combinations(labels.keys(), 2):

    # the other two
    k,l = np.setdiff1d(list(labels.keys()), (i,j))

    # already seen
    if (k,l) in dict_2.keys():
        continue

    this_mgc = copy.deepcopy(edge_mgc)
    this_mgc.merge_groups(i,j)
    # new (merged i,j) group is pushed to the back - first two must now 
    #.  be the *unmerged* remaining groups. merge them.
    this_mgc.merge_groups(0,1)

    perf, err, reg = get_reg(full_feat_vec, energies, this_mgc)

    dict_2[(i,j)] = np.mean(err**2) 
    cv_dict_2[(i,j)] = np.mean(perf)

    print("\n merged [{} and {}] in one category, [{} and {}] in the other".format(labels[i], labels[j], labels[k], labels[l]))
    print(" perf: {:.2f}  cv: {:.2f}  reg: {}".format(np.mean(err**2), np.mean(perf), reg.coef_))


for (i,j,k) in itertools.combinations(labels.keys(), 3):
    
    l = np.setdiff1d(list(labels.keys()), (i,j,k)).item()

    this_mgc = copy.deepcopy(edge_mgc)
    this_mgc.merge_groups(i,j,k)

    perf, err, reg = get_reg(full_feat_vec, energies, this_mgc)

    dict_2[(i,j,k)] = np.mean(err**2)
    cv_dict_2[(i,j,k)] = np.mean(perf)

    print("\n merged [{}, {}, and {}] in one category, [{}] in the other".format(labels[i], labels[j], labels[k], labels[l]))
    print(" perf: {:.2f}  cv: {:.2f}  reg: {}".format(np.mean(err**2), np.mean(perf), reg.coef_))


## Plot patch with node types indicated ##
###########################################

plt.close('all')

new_state = State(np.delete(state.pt_idx, [6,7,12,15,20]))
new_state.plot(mask=[0,63])

symbols = np.array(['' for i in range(state.N_tot)], dtype=object)
symbols[state.nodes_buried] = 'ko'
symbols[state.nodes_peripheral] = 'yP'

styles = np.array(['' for i in range(state.n_edges)], dtype=object)
styles[[31,51,52,54]] = ':'
styles[33] = '-.'
styles[32] = '--'
styles[63] = '-'

widths = np.array([0 for i in range(state.n_edges)])
widths[[31,32,33,51,52,54,63]] = 3

state.plot_edges(symbols=symbols, line_widths=widths, line_styles=styles)

plt.savefig("{}/Desktop/fig_nodetype".format(homedir), transparent=True)


## Do the legend ##
plt.close('all')
plt.plot([0,0], [0,0], 'ko', markersize=20, label='buried')
plt.plot([0,0], [0,0], 'yP', markersize=20, label='peripheral')
plt.plot([0,0], [0,0], 'rX', markersize=20, label='external')

plt.xlim(100,200)
plt.legend(loc='center')

plt.axis('off')
plt.savefig("{}/Desktop/leg".format(homedir), transparent=True)

## Do the legend ##
plt.close('all')
plt.plot([0,0], [0,0], 'k:', linewidth=3, label='    edge')
plt.plot([0,0], [0,0], 'k-.', linewidth=3, label='    edge')
plt.plot([0,0], [0,0], 'k--', linewidth=3, label='    edge')
plt.plot([0,0], [0,0], 'k-', linewidth=3, label='    edge')

plt.xlim(100,200)
plt.legend(loc='center')

plt.axis('off')
plt.savefig("{}/Desktop/leg_edge".format(homedir), transparent=True)


