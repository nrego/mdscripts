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


#order = ['hydrophobin', 'capsid', 'MDM2', 'malate\ndehydrogenase', 'ubiquitin', 'barnase', 'lysozyme']
order = ['2b97', '1qgt', '1ycr', '3hhp', '1ubq', '1brs', '3hhp']
colors = cm.rainbow(np.linspace(0,1,len(order)))

indices = np.arange(len(order))
fig, ax = plt.subplots(figsize=(6.5,6))
for idx, name in enumerate(order):
    dat = np.loadtxt('{}.dat'.format(name))
    arg_max = np.argmax(dat[:,2])

    phistar = dat[arg_max, 0]

    ax.bar(idx, phistar, color=colors[idx], width=1)

ax.set_xticks([])
fig.tight_layout()
fig.savefig('/Users/nickrego/Desktop/fig_s_compare.pdf', transparent=True)
plt.close('all')