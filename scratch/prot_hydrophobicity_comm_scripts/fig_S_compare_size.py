from __future__ import division, print_function
  

import MDAnalysis
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import numpy as np

import glob, os, fnmatch

import scipy.stats

## Fig S1: How surface hydrophobicity differs if you consider surface residues or atoms

mpl.rcParams.update({'axes.labelsize': 50})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 40})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':10})

# Our systems from static INDUS
fnames = glob.glob('img/*/surf_dat.dat')
labels = []
vals = np.zeros((len(fnames), 14), dtype=float)

# Load our dataset
for i, fname in enumerate(fnames):
    dirname = os.path.basename(os.path.dirname(fname))
    labels.append(dirname)
    vals[i,...] = np.loadtxt(fname)
labels = np.array(labels)

name_lup = {'1brs': 'barnase',
            '1ubq': 'ubiquitin',
            '1qgt': 'capsid',
            '1ubq': 'ubiquitin',
            '1ycr': 'MDM2',
            '253l': 'lysozyme',
            '2b97': 'hydrophobin',
            '3hhp': 'malate\ndehydrogenase'}

order = ['hydrophobin', 'capsid', 'MDM2', 'malate\ndehydrogenase', 'ubiquitin', 'barnase', 'lysozyme']

colors = cm.rainbow(np.linspace(0,1,len(order)))

order_idx = np.zeros(len(order), dtype=int)
for idx, label in enumerate(labels):
    if name_lup[label] in order:
        order_idx[order.index(name_lup[label])] = idx

vals = vals[order_idx,...]
labels = labels[order_idx]

# Colors in dictionary by label
color_lup = {k:v for k,v in zip(labels, colors)}


# Gather new data (random sample) and sort 
new_fnames = glob.glob('*/surf_dat.dat')
new_labels = []
for fname in new_fnames:
    dirname = os.path.dirname(fname)
    if dirname not in ['1wr1_ubiq', '2k6d_ubiq', '2qho_ubiq', '2z59_ubiq', '1bmd', '2b97', '1qgt']:
        new_labels.append(dirname)

new_labels = np.array(new_labels)
new_vals = np.zeros((new_labels.size, 14))

for i, new_label in enumerate(new_labels):
    fname = '{}/surf_dat.dat'.format(new_label)
    new_vals[i,:] = np.loadtxt(fname)


# Combine new and old values and labels
vals = np.vstack((vals, new_vals))
labels = np.append(labels, new_labels)

# Sort by number of surface atoms
sort_idx = np.argsort(vals[:,1])
vals_atm = vals[sort_idx,1]
labels_atm = labels[sort_idx]

# Sort by number of surface residues
sort_idx = np.argsort(vals[:,2])
vals_res = vals[sort_idx,2]
labels_res = labels[sort_idx]

# Sort by number of surface heavy atoms
sort_idx = np.argsort(vals[:,8])
vals_atm_h = vals[sort_idx, 8]
labels_atm_h = labels[sort_idx]


indices = np.arange(labels.size)

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3,figsize=(12,7))
width = 1
n_bar = indices.size


for idx in indices:
    label_atm = labels_atm[idx]
    label_res = labels_res[idx]
    label_atm_h = labels_atm_h[idx]

    val_atm = vals_atm[idx]
    val_res = vals_res[idx]
    val_atm_h = vals_atm_h[idx]

    label = labels[idx]

    if label_atm in color_lup.keys():
        this_color_atm = color_lup[label_atm]
    else:
        this_color_atm = '#8C8D8F'

    if label_res in color_lup.keys():
        this_color_res = color_lup[label_res]
    else:
        this_color_res = '#8C8D8F'

    if label_atm_h in color_lup.keys():
        this_color_atm_h = color_lup[label_atm_h]
    else:
        this_color_atm_h = '#8C8D8F'

    if label in color_lup.keys():
        this_color_pt = color_lup[label]
        this_order = 3
        this_size = 10
    else:
        this_color_pt = '#8C8D8F'
        this_order = 1
        this_size = 8


    ax1.bar(idx, val_atm, width=width, color=this_color_atm, label=r'${}$'.format(label_atm))
    ax2.bar(idx, val_res, width=width, color=this_color_res, label=r'${}$'.format(label_res))
    ax3.bar(idx, val_atm_h, width=width, color=this_color_atm_h, label=r'${}$'.format(label_atm_h))

    

    ax5.plot(vals[idx, 2], vals[idx, 1], 'o', color=this_color_pt, zorder=this_order, markersize=this_size)
    ax6.plot(vals[idx,8], vals[idx,1], 'o', color=this_color_pt, zorder=this_order, markersize=this_size)


xvals = np.arange(0,2000,1)
# Regress # surface atoms on # surface residues
slope, inter, rval, pval, std = scipy.stats.linregress(vals[:,2], vals[:,1])
ax5.plot(xvals, inter+slope*xvals, 'k-', zorder=0, linewidth=2)
# Regress # surface atoms on # surface heavy atoms
slope, inter, rval, pval, std = scipy.stats.linregress(vals[:,8], vals[:,1])
ax6.plot(xvals, inter+slope*xvals, 'k-', zorder=0, linewidth=2)

ax4.axis('off')
ax1.set_xticks([])
ax1.set_ylim(0,3000)
ax2.set_xticks([])
ax2.set_ylim(0,250)
ax3.set_xticks([])
ax3.set_ylim(0,1500)

ax5.set_xlim(0,250)
ax5.set_ylim(0,3000)
ax6.set_xlim(0,1500)
ax6.set_ylim(0,3000)
ax6.set_yticks([])

fig.tight_layout()
fig.subplots_adjust(hspace=0.5)


fig.savefig('/Users/nickrego/Desktop/fig.pdf', transparent=True)

