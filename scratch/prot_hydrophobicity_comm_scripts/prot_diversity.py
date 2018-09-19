from __future__ import division, print_function
  

import MDAnalysis
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import numpy as np

import glob, os

mpl.rcParams.update({'axes.labelsize': 50})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 80})
mpl.rcParams.update({'axes.titlesize': 50})
mpl.rcParams.update({'legend.fontsize':40})

name_lup = {'1brs': 'barnase',
            '1ubq': 'ubiquitin',
            '1qgt': 'capsid',
            '1ycr': 'MDM2',
            '253l': 'lysozyme',
            '2b97': 'hydrophobin',
            '3hhp': 'malate dehydrogenase'}

order = ['hydrophobin', 'capsid', 'MDM2', 'malate dehydrogenase', 'barnase', 'lysozyme']

colors = cm.rainbow(np.linspace(0,1,len(order)))

dat = np.loadtxt('surf_dat.dat', dtype=object)
labels = dat[:,0]
vals = dat[:,1:].astype(float)

order_idx = np.zeros(len(order), dtype=int)
for idx, label in enumerate(labels):
    if name_lup[label] in order:
        order_idx[order.index(name_lup[label])] = idx

vals = vals[order_idx,...]
labels = labels[order_idx]

names = [name_lup[label] for label in labels]

fig, ax = plt.subplots(figsize=(10,8))
n_bars = len(labels)
indices = np.arange(n_bars)
width = 1

for idx, label in enumerate(names):
    ax.bar(indices[idx], vals[idx, 2], width=width, label=label, color=colors[idx])
    
ax.set_xticks([])
ax.set_yticks(np.arange(0.0, 0.70,0.1))
ax.set_ylim(0.1, 0.35)
#ax.set_ylim(3000,4000)
#ax.legend(handlelength=1)

plt.tight_layout()
plt.savefig('/Users/nickrego/Desktop/n_charge.png', transparent=True)
plt.show()

