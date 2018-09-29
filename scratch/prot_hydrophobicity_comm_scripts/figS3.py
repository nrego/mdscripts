from __future__ import division, print_function
  

import MDAnalysis
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import numpy as np

import glob, os, fnmatch


## Fig S1: How surface hydrophobicity differs if you consider surface residues or atoms

mpl.rcParams.update({'axes.labelsize': 50})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 40})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':10})

# Our systems from static INDUS
fnames = glob.glob('img/*/surf_dat.dat')
labels = []
vals = np.zeros((len(fnames), 8), dtype=float)

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

color_lup = {k:v for k,v in zip(labels, colors)}


# Gather all data and sort
sample_vals_phob = vals[:,4] / vals[:,1]
sample_vals_charge = (vals[:,6]+vals[:,7])/vals[:,1]

new_fnames = glob.glob('*/surf_dat.dat')
new_vals_phob = []
new_vals_charge = []
new_labels = []

for fname in new_fnames:
    dirname = os.path.dirname(fname)
    if dirname not in ['1wr1_ubiq', '2k6d_ubiq', '2qho_ubiq', '2z59_ubiq', '1bmd', '2b97', '1qgt']:
        new_labels.append(dirname)
        dat = np.loadtxt(fname)
        new_vals_phob.append(dat[4]/dat[1])
        new_vals_charge.append((dat[6]+dat[7])/dat[1])

vals_phob = np.append(sample_vals_phob, new_vals_phob)
vals_charge = np.append(sample_vals_charge, new_vals_charge)
labels = np.append(labels, new_labels)

# Sort the hydrophobicity
sort_idx = np.argsort(vals_phob)
vals_phob = vals_phob[sort_idx]
labels_phob = labels[sort_idx]

# Sort charge density
sort_idx = np.argsort(vals_charge)
vals_charge = vals_charge[sort_idx]
labels_charge = labels[sort_idx]

indices = np.arange(vals_phob.size)

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(18,6.5))
width = 1
n_bar = indices.size
gap = 4
# upper left
for idx in indices:
    label_phob = labels_phob[idx]
    label_charge = labels_charge[idx]

    this_val_phob  = vals_phob[idx] 
    this_val_charge = vals_charge[idx]

    if label_phob in color_lup.keys():
        this_color_phob = color_lup[label_phob]
    else:
        this_color_phob = '#8C8D8F'

    if label_charge in color_lup.keys():
        this_color_charge = color_lup[label_charge]
    else:
        this_color_charge = '#8C8D8F'


    ax1.bar(idx, this_val_phob, width=width, color=this_color_phob, label=r'${}$'.format(label_phob))
    ax2.bar(idx+n_bar+gap, this_val_charge, width=width, color=this_color_charge, label=r'${}$'.format(label_charge))

    ax1.set_xticks([])
    ax1.set_ylim(0.5, 0.66)

    ax2.set_xticks([])
    ax2.set_ylim(0.01, 0.03)
    ax2.set_yticks([0.01,0.02,0.03])

fig.tight_layout()
fig.subplots_adjust(wspace=1.0)


fig.savefig('/Users/nickrego/Desktop/fig.pdf', transparent=True)


