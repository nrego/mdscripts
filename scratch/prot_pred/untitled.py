import glob, os, sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

from mdtools import dr

mpl.rcParams.update({'axes.labelsize': 50})
mpl.rcParams.update({'xtick.labelsize': 20})
mpl.rcParams.update({'ytick.labelsize': 20})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':20})


labels = ['1msb', '1gvp', '1hjr', '1msb', '1qgt', '1ycr', '2b97', '2fi2', '2qho', 'ubiq']
fnames = glob.glob("*/pred/corr_phi.dat")

phi_stars = []
min_ds = []
for fname in fnames:
    this_phistar, this_min_d = np.loadtxt(fname)

    phi_stars = np.append(phi_stars, this_phistar)
    min_ds = np.append(min_ds, this_min_d)

indices = np.arange(min_ds.size)
width = 0.35

fig, ax = plt.subplots(figsize=(10,8))

ax.bar(indices, phi_stars, width=width, label=r'peak susceptibility ($\phi^*$)')
ax.bar(indices+width, min_ds, width=width, label='optimal interface')

ax.set_xticks(indices+(width/2.0))
ax.set_xticklabels(labels, rotation=45)
ax.set_ylabel(r'$\phi \; \rm{kJ/mol}$')
ax.legend()
ax.set_ylim(0,7)

fig.tight_layout()
fig.savefig('/Users/nickrego/Desktop/fig.png')