from __future__ import division, print_function

import numpy as np
import matplotlib
mpl = matplotlib
from matplotlib import pyplot as plt
from matplotlib.image import NonUniformImage, imread
from scipy.optimize import minimize


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from mdtools import dr

from constants import k

import os
import glob

from mdtools import dr

beta = 1/(k*300)


mpl.rcParams.update({'axes.labelsize': 30})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize': 30})


fnames = glob.glob('mu_*')

alphas = np.array([])
avg_n = np.array([])
avg_ntwid = np.array([])

for fname in fnames:
    this_alpha = float(fname.split('_')[-1]) / 10.0

    alphas = np.append(alphas, this_alpha)

    dat = np.loadtxt('{}/averages.dat'.format(fname))
    avg_n = np.append(avg_n, dat[0])
    avg_ntwid = np.append(avg_ntwid, dat[1])


fig, ax = plt.subplots(figsize=(5,4))
ax.plot(alphas*beta, avg_ntwid, '-ok', linewidth=6, markersize=12)
indices = [0, 7, 14]
ax.plot(alphas[indices]*beta, avg_ntwid[indices], 'or', markersize=12)
ax.set_xticks([0,2,4,6,8])
ax.set_xlabel(r'$\beta \alpha$')
ax.set_ylabel(r'$\langle \tilde{N}_v \rangle_\alpha$')
fig.tight_layout()

fig.savefig('/Users/nickrego/Desktop/fig.pdf', transparent=True)


