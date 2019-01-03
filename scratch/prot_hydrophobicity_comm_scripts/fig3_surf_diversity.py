from __future__ import division, print_function
  

import MDAnalysis
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import numpy as np

import glob, os

mpl.rcParams.update({'axes.labelsize': 50})
mpl.rcParams.update({'xtick.labelsize': 60})
mpl.rcParams.update({'ytick.labelsize': 60})
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


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15,14), sharex=True)

# upper left
for idx, name in enumerate(names):
    label = labels[idx]
    this_vals = vals[idx] 
    n_surf = this_vals[1]
    n_res_surf = this_vals[2]
    n_phob = this_vals[4]
    n_phob_res = this_vals[5]
    n_pos_charge = this_vals[6]
    n_neg_charge = this_vals[7]

    this_dipole = dipole[label]

    # Fraction charged surface residues
    ax1.bar(idx, (n_pos_charge+n_neg_charge)/n_res_surf, width=width, label=name, color=colors[idx])

    ax2.axis('off')

    # Fraction hydrophobic surface residues
    ax3.bar(idx, n_phob_res/n_res_surf, width=width, label=name, color=colors[idx])

    # Fraction hydrophobic surface atoms
    ax4.bar(idx, n_phob/n_surf, width=width, label=name, color=colors[idx])

yticks = np.arange(0, 1.1, 0.1)
ax1.set_xticks([])
ax1.set_yticks(yticks)
ax1.set_ylim(0, 0.4)

ax3.set_xticks([])
ax3.set_yticks(yticks)
ax3.set_ylim(0, 0.75)

ax4.set_xticks([])
ax4.set_yticks(yticks)
ax4.set_ylim(0, 0.75)

fig.tight_layout()
fig.subplots_adjust(hspace=0.3)
fig.savefig('/Users/nickrego/Desktop/blah.pdf', transparent=True)
#plt.show()
plt.clf()
fig, ax = plt.subplots(figsize=(10,10))
for idx, name in enumerate(names):
    label = labels[idx]

    ax.bar(indices[idx], 0, width=width, label=name, color=colors[idx])

ax.set_ylim(10,12)
ax.set_xticks([])
ax.set_yticks([])



plt.legend(handlelength=0.5, labelspacing=0.3, framealpha=0.0)
plt.tight_layout()
plt.savefig('/Users/nickrego/Desktop/label.pdf', transparent=True)

