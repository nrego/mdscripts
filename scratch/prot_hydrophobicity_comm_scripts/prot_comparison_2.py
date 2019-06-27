from __future__ import division, print_function
  

import MDAnalysis
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

import glob, os

mpl.rcParams.update({'axes.labelsize': 50})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize': 50})
mpl.rcParams.update({'legend.fontsize':20})

name_lup = {'1brs': 'barnase',
            '1ubq': 'ubiquitin',
            '1qgt': 'capsid',
            '1ycr': 'mdm2',
            '253l': 'lysozyme',
            '2b97': 'hydrophobin',
            '3hhp': 'malate dehydrogenase'}

order = ['hydrophobin', 'ubiquitin', 'capsid', 'lysozyme', 'mdm2', 'malate dehydrogenase', 'barnase']

from constants import k
indices = np.arange(1)
width = 0.2

fig, ax = plt.subplots()

order_idx = []
fnames = np.array([], dtype=str)
for key, val in name_lup.items():
    if val in order:
        order_idx.append(order.index(val))
        fnames = np.append(fnames, '{}/struct_data.dat'.format(key))
fnames = fnames[np.argsort(order_idx)]

for idir, fname in enumerate(fnames):

    dirname = os.path.dirname(fname)
    try:
        dat = np.loadtxt(fname)
        n_heavy = dat[3]
        n_surf = dat[4]
        n_surf_hydrophil = dat[5]
        n_surf_hydrophob = dat[6]

        frac_surf = n_surf / n_heavy
        frac_surf_hydrophil = n_surf_hydrophil / n_surf
        frac_surf_hydrophob = n_surf_hydrophob / n_surf

        ax.bar(indices+idir*width, [frac_surf], width=width, label=order[idir])
    except:
        pass

ax.set_ylabel('Fraction of atoms on surface')
#ax.set_xticks(indices+width*3.0)
ax.set_xticks([])
#ax.set_xticklabels(('Heavy Atoms', 'Surface Heavy Atoms'))

ax.legend()
fig.tight_layout()

plt.show()

