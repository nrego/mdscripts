from __future__ import division, print_function
  

import MDAnalysis
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import numpy as np

import glob, os, fnmatch


## Fig S2: How surface chemistries (frac hydrophobic surface atoms, frac hydrophobic surface residues)
#  change between proteins

mpl.rcParams.update({'axes.labelsize': 50})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 40})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':10})

# Our systems from static INDUS
fnames = glob.glob('img/*/surf_dat.dat')
labels = []
vals = np.zeros((len(fnames), 14), dtype=float)

# Load our core datasets
for i, fname in enumerate(fnames):
    dirname = os.path.basename(os.path.dirname(fname))
    labels.append(dirname)
    vals[i,...] = np.loadtxt(fname)
labels = np.array(labels)

# Arange their colors according to rainbow in below order...
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

color_lup = {k:v for k,v in zip(labels, colors)}


# Gather new data (random sample)
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


# Add new colors (all grey)
for label in new_labels:
    assert label not in color_lup.keys()
    color_lup[label] = '#8C8D8F'

# Combine new and old values and labels
vals = np.vstack((vals, new_vals))
labels = np.append(labels, new_labels)

n_atm = vals[:,1]
n_atm_h = vals[:,8]
n_res = vals[:,2]

n_phob_atm = vals[:,4]
n_phob_atm_h = vals[:,9]
n_phob_res = vals[:,5]
n_charge_res = vals[:,6] + vals[:,7]

# Sort by fraction charged residues
sort_idx = np.argsort(n_charge_res/n_res)
vals_charge_res = (n_charge_res/n_res)[sort_idx]
labels_charge_res = labels[sort_idx]

# Sort by fraction hydrophobic residues
sort_idx = np.argsort(n_phob_res/n_res)
vals_phob_res = (n_phob_res/n_res)[sort_idx]
labels_phob_res = labels[sort_idx]

# Sort by fraction hydrophobic atoms
sort_idx = np.argsort(n_phob_atm/n_atm)
vals_phob_atm = (n_phob_atm/n_atm)[sort_idx]
labels_phob_atm = labels[sort_idx]

# Sort by fraction hydrophobic heavy atoms
sort_idx = np.argsort(n_phob_atm_h/n_atm_h)
vals_phob_atm_h = (n_phob_atm_h/n_atm_h)[sort_idx]
labels_phob_atm_h = labels[sort_idx]


indices = np.arange(labels.size)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(16,12))
width = 1
n_bar = indices.size


for idx in indices:
    label_charge_res = labels_charge_res[idx]
    label_phob_res = labels_phob_res[idx]
    label_phob_atm = labels_phob_atm[idx]
    label_phob_atm_h = labels_phob_atm_h[idx]

    val_charge_res = vals_charge_res[idx]
    val_phob_res = vals_phob_res[idx]
    val_phob_atm = vals_phob_atm[idx]
    val_phob_atm_h = vals_phob_atm_h[idx]

    this_color_charge_res = color_lup[label_charge_res]
    this_color_phob_res = color_lup[label_phob_res]
    this_color_phob_atm = color_lup[label_phob_atm]
    this_color_phob_atm_h = color_lup[label_phob_atm_h]


    ax1.bar(idx, val_charge_res, width=width, color=this_color_charge_res, label=r'${}$'.format(label_charge_res))
    ax2.bar(idx, val_phob_res, width=width, color=this_color_phob_res, label=r'${}$'.format(label_phob_res))
    ax3.bar(idx, val_phob_atm_h, width=width, color=this_color_phob_atm_h, label=r'${}$'.format(label_phob_atm_h))
    ax4.bar(idx, val_phob_atm, width=width, color=this_color_phob_atm, label=r'${}$'.format(label_phob_atm))

    if idx == 0 or idx == n_bar-1:
        ax1.axhline(y=val_charge_res, linestyle='--', color='k')
        ax2.axhline(y=val_phob_res, linestyle='--', color='k')
        ax3.axhline(y=val_phob_atm_h, linestyle='--', color='k')
        ax4.axhline(y=val_phob_atm, linestyle='--', color='k')

ticks = np.arange(0,1.1,0.1)

ax1.set_xticks([])
ax1.set_yticks(ticks)
ax1.set_ylim(0,0.7)

ax2.set_xticks([])
ax2.set_yticks(ticks)
ax2.set_ylim(0,0.7)

ax3.set_xticks([])
ax3.set_yticks(ticks)
ax3.set_ylim(0,0.7)

ax4.set_xticks([])
ax4.set_yticks(ticks)
ax4.set_ylim(0,0.7)

fig.tight_layout()
fig.subplots_adjust(hspace=0.5)


fig.savefig('/Users/nickrego/Desktop/fig.pdf', transparent=True)

