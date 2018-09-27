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
            '1ycr': 'MDM2',
            '253l': 'lysozyme',
            '2b97': 'hydrophobin',
            '3hhp': 'malate dehydrogenase'}

order = ['hydrophobin', 'capsid', 'MDM2', 'malate dehydrogenase', 'barnase', 'lysozyme']

colors = cm.rainbow(np.linspace(0,1,len(order)))

order_idx = np.zeros(len(order), dtype=int)
for idx, label in enumerate(labels):
    if name_lup[label] in order:
        order_idx[order.index(name_lup[label])] = idx

vals = vals[order_idx,...]
labels = labels[order_idx]

names = [name_lup[label] for label in labels]
dipole = np.loadtxt('dipole.dat', usecols=(1,))

assert np.array_equal(labels, np.loadtxt('dipole.dat', usecols=(0,), dtype='|S4'))

fig, ax = plt.subplots(figsize=(10,8))
n_bars = len(labels)
indices = np.arange(n_bars)
width = 1

gap = 4
for idx, label in enumerate(names):
    n_tot, n_surf, n_res_surf, n_phil_surf, n_phob_surf, n_phob_res, pos_charge_res, neg_charge_res  = vals[idx] 
    #ax.bar(indices[idx], n_phob_res/n_res_surf, width=width, label=label, color=colors[idx])
    #ax.bar(indices[idx]+n_bars+gap, n_phob_surf/n_surf, width=width, label=label, color=colors[idx])
    ax.bar(indices[idx], (pos_charge_res+neg_charge_res)/n_surf, width=width, label=label, color=colors[idx])

ax.set_xlim(-1.3, 6.3)
#ax.set_xticks([2.5, 2.5+n_bars+gap])
#ax.set_xticklabels(('residues', 'atoms'))
ax.set_xticks([])
#ax.set_yticks(np.arange(0.0, 0.70,0.1))
ax.set_ylim(0.01, 0.027)
#ax.set_ylim(3000,4000)
#ax.legend(handlelength=1)
ax.set_title('Charge')
plt.tight_layout()
plt.savefig('/Users/nickrego/Desktop/charge.pdf', transparent=True)
plt.show()

