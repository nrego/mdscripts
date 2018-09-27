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
vals = np.zeros((len(fnames), 8), dtype=float)

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
            '3hhp': 'malate dehydrogenase'}

order = ['hydrophobin', 'capsid', 'MDM2', 'malate dehydrogenase', 'ubiquitin', 'barnase', 'lysozyme']

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
    n_tot, n_surf, n_res_surf, n_phil_surf, n_phob_surf, n_phob_res, pos_charge_res, neg_charge_res  = vals[idx] 
    this_dipole = dipole[label]

    ax1.bar(indices[idx], n_phob_res/n_res_surf, width=width, label=name, color=colors[idx])
    ax2.bar(indices[idx], n_phob_surf/n_surf, width=width, color=colors[idx])
    ax1.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6])
    ax1.set_ylim(0.2, 0.66)
    ax1.set_xticks([])
    ax2.set_yticks([])
    ax2.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6])
    ax2.set_ylim(0.2, 0.66)
    ax2.set_xticks([])
    
    ax3.bar(indices[idx], (pos_charge_res+neg_charge_res)/n_surf, width=width, label=label, color=colors[idx])
    ax4.bar(indices[idx], this_dipole/n_surf, width=width, label=label, color=colors[idx])
    ax3.set_xticks([])
    ax4.set_xticks([])

fig.tight_layout()
fig.subplots_adjust(hspace=0.3)
fig.savefig('/Users/nickrego/Desktop/blah.pdf', transparent=True)
#plt.show()
plt.clf()
fig, ax = plt.subplots(figsize=(10,10))
for idx, name in enumerate(names):
    label = labels[idx]
    n_tot, n_surf, n_res_surf, n_phil_surf, n_phob_surf, n_phob_res, pos_charge_res, neg_charge_res  = vals[idx] 
    this_dipole = dipole[label]

    ax.bar(indices[idx], n_phob_res/n_res_surf, width=width, label=name, color=colors[idx])

    ax.set_ylim(10,12)
    ax.set_xticks([])
    ax.set_yticks([])



plt.legend(handlelength=1, labelspacing=0.1)
plt.tight_layout()
plt.savefig('/Users/nickrego/Desktop/label.pdf', transparent=True)

