from __future__ import division, print_function

import numpy as np
import glob, os

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
from IPython import embed

import networkx as nx

from scipy.spatial import cKDTree

# 'reg' or 'inv'
mode = 'inv'
## From pattern_sample directory, extract free energies for each k, d value
mpl.rcParams.update({'axes.labelsize': 20})
mpl.rcParams.update({'xtick.labelsize': 10})
mpl.rcParams.update({'ytick.labelsize': 10})
mpl.rcParams.update({'axes.titlesize': 30})

class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]

        return np.ma.masked_array(np.interp(value, x, y))

def gen_graph(positions, indices):

    if indices.size == 0:
        return nx.Graph, None

    if indices.ndim == 0:
        indices = indices[None]

    local_indices = np.arange(indices.size)
    these_positions = positions[indices]

    tree = cKDTree(these_positions)
    local_edges = list(tree.query_pairs(r=0.6))

    graph = nx.Graph()
    graph.add_nodes_from(indices)

    for local_i, local_j in local_edges:
        global_i = indices[local_i]
        global_j = indices[local_j]
        graph.add_edge(global_i, global_j)

    pos_dict = {}
    for idx, pos in zip(indices, these_positions):
        pos_dict[idx] = pos

    return graph, pos_dict

max_val = 1.75
rms_bins = np.arange(0, max_val+0.1, 0.05)
homedir = os.environ['HOME']


# beta mu_ex=-log(P_v(0)) in bulk
ref_mu = np.loadtxt('{}/simulations/pattern_sample/bulk_00/PvN.dat'.format(homedir))[0,1]
positions = np.loadtxt('positions.dat')
fig, ax = plt.subplots(figsize=(5.5,6))
ax.set_xticks([])
ax.set_yticks([])
ax.plot(positions[:,0], positions[:,1], 'o', markeredgecolor='k', markeredgewidth=2, markerfacecolor='w', markersize=24)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for idx in range(36):
    ax.annotate(idx, xy=positions[idx]-0.025)
fig.savefig('positions_labeled.pdf')
plt.close('all')

if mode == 'reg':
    fnames = glob.glob('k_*/d_*/trial_0/PvN.dat') + ['k_36/PvN.dat', 'k_00/PvN.dat']
else:
    fnames = glob.glob('l_*/d_*/trial_0/PvN.dat')

k_bins = np.arange(0,38,1)

energies = np.zeros((rms_bins.size-1, k_bins.size-1))
energies[:] = np.nan
errors = np.zeros((rms_bins.size-1, k_bins.size-1))
errors[:] = np.nan

dg_bind = np.zeros_like(energies)
#dg_bind[:] = np.nan

range_k = np.zeros(k_bins.size-1)

n_clust = np.zeros((rms_bins.size-1, k_bins.size-1, 2))
n_clust[:] = np.nan

methyl_masks = np.zeros((rms_bins.size-1, k_bins.size-1, 36), dtype=bool)

for fname in fnames:
    print("fname: {}".format(fname))

    subdir = os.path.dirname(fname)
    this_k = int(fname.split('/')[0].split('_')[-1])
    
    if this_k == 36:
        this_d = 1.14
    elif this_k == 0:
        this_d = 0
    else:
        this_d = float(fname.split('/')[1].split('_')[-1]) / 100

    idx_k = np.digitize(this_k, k_bins) - 1
    idx_d = np.digitize(this_d+0.001, rms_bins) - 1

    this_fvn = np.loadtxt(fname)

    this_mu, this_err = this_fvn[0, 1:]

    energies[idx_d, idx_k] = this_mu
    errors[idx_d, idx_k] = this_err

    dg_bind[idx_d, idx_k] = this_mu - ref_mu

    
    # Find the graph of methyl and oh groups
    indices_ch3 = np.loadtxt('{}/this_pt.dat'.format(subdir), dtype=int, ndmin=1)
    indices_oh = np.setdiff1d(np.arange(36), indices_ch3)
    methyl_mask = np.zeros(36, dtype=bool)
    methyl_mask[indices_ch3] = True
    if mode == 'reg':
        assert methyl_mask.sum() == this_k
    else:
        assert methyl_mask.sum() == 36 - this_k

    methyl_masks[idx_d, idx_k] = methyl_mask

    '''
    graph_ch3, pos_ch3 = gen_graph(positions, indices_ch3)
    graph_oh, pos_oh = gen_graph(positions, indices_oh)

    n_clust_ch3 = len(list(nx.connected_components(graph_ch3)))
    n_clust_oh = len(list(nx.connected_components(graph_oh)))

    n_clust[idx_d, idx_k, 0] = n_clust_ch3
    n_clust[idx_d, idx_k, 1] = n_clust_oh

    if graph_ch3.number_of_nodes:
        fig, ax = plt.subplots(figsize=(5.5,6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        nx.draw(graph_ch3, with_labels=True, ax=ax, pos=pos_ch3, node_color='gray', markersize=18)
        fig.savefig('{}/graph_ch3.pdf'.format(subdir))
        plt.close('all')

    if graph_ch3.number_of_nodes:
        fig, ax = plt.subplots(figsize=(5.5,6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        nx.draw(graph_oh, with_labels=True, ax=ax, pos=pos_oh, node_color='blue', markersize=18)
        fig.savefig('{}/graph_oh.pdf'.format(subdir))
        plt.close('all')
    '''

for idx_k in range(k_bins.size-1):
    range_k[idx_k] = np.nanmax(energies[:,idx_k]) - np.nanmin(energies[:,idx_k])

fig, ax = plt.subplots(figsize=(6,5))
ax.bar(k_bins[:-1], range_k, width=0.8, align='edge')
ax.set_ylabel(r'$\Delta \beta F$')
ax.set_xlabel(r'$k$')
fig.savefig('range_k.pdf')
plt.close('all')

fig, ax = plt.subplots(figsize=(6,5))

min_energy = np.nanmin(energies)
max_energy = np.nanmax(energies)
norm = mpl.colors.Normalize(vmin=min_energy, vmax=max_energy)

extent = (rms_bins[0], rms_bins[-1], k_bins[0], k_bins[-1])
im = ax.imshow(energies.T, extent=extent, origin='lower', aspect='auto', norm=norm, cmap=cm.nipy_spectral)
cb = plt.colorbar(im)
#ax.set_xlabel(r'$d_\mathrm{CH3}$ (RMS)')
#ax.set_ylabel(r'$k$')

fig.tight_layout()
fig.savefig('rms_k_2d.pdf')

fig, ax = plt.subplots(figsize=(6,5))

min_err = np.nanmin(errors)
max_err = np.nanmax(errors)
norm = mpl.colors.Normalize(vmin=min_err, vmax=max_err)

extent = (rms_bins[0], rms_bins[-1], k_bins[0], k_bins[-1])
im = ax.imshow(errors.T, extent=extent, origin='lower', aspect='auto', norm=norm, cmap=cm.nipy_spectral)
cb = plt.colorbar(im)

fig.tight_layout()
fig.savefig('err.pdf')

np.savez_compressed('analysis_data.dat', energies=energies, methyl_mask=methyl_masks, positions=positions, rms_bins=rms_bins, k_bins=k_bins)

## Plot dg_bind ##


fig, ax = plt.subplots(figsize=(6,5))
min_dg = np.nanmin(dg_bind)
max_dg = np.nanmax(dg_bind)
dg_lim = np.abs(min_dg)
assert np.abs(min_dg) > np.abs(max_dg)
#norm = mpl.colors.Normalize(vmin=-dg_lim, vmax=dg_lim)
norm = MidpointNormalize(vmin=min_dg, vmax=max_dg, midpoint=0)

extent = (rms_bins[0], rms_bins[-1], k_bins[0], k_bins[-1])
im = ax.imshow(dg_bind.T, extent=extent, origin='lower', aspect='auto', norm=norm, cmap=cm.seismic_r)
cb = plt.colorbar(im)
#ax.set_xlabel(r'$d_\mathrm{CH3}$ (RMS)')
#ax.set_ylabel(r'$k$')
#ax.set_clim(min_dg, max_dg)

fig.tight_layout()
fig.savefig('dg_bind_k_2d.pdf')

'''
#########################
# Number of clusters
#########################
all_ch3_clust = n_clust[:,:,0]
all_oh_clust = n_clust[:,:,1]



min_clust = np.nanmin(n_clust)
max_clust = np.nanmax(n_clust)
norm = mpl.colors.Normalize(vmin=min_clust, vmax=max_clust)

extent = (rms_bins[0], rms_bins[-1], k_bins[0], k_bins[-1])
cmap = cm.get_cmap('jet', max_clust-min_clust)
##############
# N CH3 clust
##############

fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(all_ch3_clust.T, extent=extent, origin='lower', aspect='auto', norm=norm, cmap=cmap)
cb = plt.colorbar(im)
ax.set_xlabel(r'$d_\mathrm{CH3}$ (RMS)')
ax.set_ylabel(r'$k$')

fig.tight_layout()
fig.savefig('n_ch3_clust.pdf')

##############
# N OH clust
##############

fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(all_oh_clust.T, extent=extent, origin='lower', aspect='auto', norm=norm, cmap=cmap)
cb = plt.colorbar(im)
ax.set_xlabel(r'$d_\mathrm{CH3}$ (RMS)')
ax.set_ylabel(r'$k$')

fig.tight_layout()
fig.savefig('n_oh_clust.pdf')
'''