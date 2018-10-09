from __future__ import division, print_function
  

import MDAnalysis
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import numpy as np

import glob, os

mpl.rcParams.update({'axes.labelsize': 50})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 40})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':40})

fnames = glob.glob('*/surf_dat.dat')
labels = []

vals = np.zeros((len(fnames), 14), dtype=float)


for i, fname in enumerate(fnames):
    dirname = os.path.dirname(fname)
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

names = [name_lup[label] for label in labels]
dipole = {k:v for k,v in zip(np.loadtxt('dipole.dat', usecols=0, dtype=str), np.loadtxt('dipole.dat', usecols=1))}

n_bars = len(labels)
indices = np.arange(n_bars)
width = 1

gap = 4


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,8), sharex=True)

# upper left
for idx, name in enumerate(names):
    label = labels[idx]
    this_vals = vals[idx] 
    n_phob = this_vals[4]
    n_phob_from_phob = this_vals[-1]
    n_phob_from_phil = this_vals[-3]

    this_dipole = dipole[label]

    # Fraction hydrophobic atoms from hydrophobic residues
    ax1.bar(idx, n_phob_from_phob/n_phob, width=width, label=name, color=colors[idx])
    ax2.bar(idx, n_phob_from_phil/n_phob, width=width, label=name, color=colors[idx])

yticks = np.arange(0, 1.1, 0.1)

ax1.set_xticks([])
ax1.set_yticks(yticks)

ax2.set_xticks([])
ax2.set_yticks(yticks)

fig.tight_layout()
fig.subplots_adjust(hspace=0.3)
fig.savefig('/Users/nickrego/Desktop/blah.pdf', transparent=True)


