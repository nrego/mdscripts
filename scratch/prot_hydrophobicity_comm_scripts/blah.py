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

order = ['2b97', '1qgt', '1ycr', '3hhp', '1ubq', '1brs', '253l']

colors = cm.rainbow(np.linspace(0,1,len(order)))

n_bars = len(order)
indices = np.arange(n_bars)
width = 1

gap = 4


fig, ax = plt.subplots(figsize=(7.5,6.5), sharex=True)

isothermal_comp = 3.467e17
rho = 33
kT = (1.3806485e-23)*(300)
wat_val = isothermal_comp*rho*kT
#ax.plot(0, wat_val, '_', markersize=20)
# upper left
for idx, name in enumerate(order):
    surf_dat = np.loadtxt('{}/surf_dat.dat'.format(name))
    n_dat = np.loadtxt('{}.dat'.format(name))

    n_surf = surf_dat[1]
    n_phob = surf_dat[4]

    if name != 'ubiq':
        fmt = '-o'
        markersize=8
    else:
        fmt = '-'
        markersize=0

    ax.plot(n_dat[:,0], n_dat[:,2]/n_dat[:,1], fmt, color=colors[idx], linewidth=3, markersize=markersize)

    max_sus = n_dat[:,2].max()
    norm_max = max_sus / n_dat[0,1]
    #ax.plot(n_phob/n_surf, norm_max, 'o')
    #ax.bar(idx, max_sus, color=colors[idx], width=width)
    #ax.plot(n_dat[0,1], max_sus, 'o')

#ax.set_xticks([])
ax.set_xlim(0,4)

fig.tight_layout()
fig.savefig('/Users/nickrego/Desktop/peak_sus.pdf', transparent=True)
#plt.show()

plt.close('all')