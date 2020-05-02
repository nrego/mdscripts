from __future__ import division

import numpy as np
import MDAnalysis

import argparse
from IPython import embed
import os, glob

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np

from scratch.sam.util import *

from scipy.special import binom


plt.close('all')
homedir = os.environ['HOME']
from IPython import embed
mpl.rcParams.update({'axes.labelsize': 45})
mpl.rcParams.update({'xtick.labelsize': 35})
mpl.rcParams.update({'ytick.labelsize': 35})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':30})


### PLOT State functions of full model from WL sampling 
###    DOS for p, q, k_o, n_oo, n_oe
###
#########################################

ds = np.load('data/sam_dos.npz')

reg = np.load('data/sam_reg_m3.npy').item()

# p,q combos we're considering
vals_pq = ds['vals_pq']

feat_pq = np.hstack((vals_pq.prod(axis=1)[:,None], vals_pq))
vals_ko = ds['vals_ko']
vals_noo = ds['vals_noo']
vals_noe = ds['vals_noe']

x_ko, x_noo, x_noe = np.meshgrid(vals_ko, vals_noo, vals_noe, indexing='ij')

# shape: (n_ko, n_noo, n_noe)
energies = reg.coef_[0]*x_ko + reg.coef_[1]*x_noo + reg.coef_[2]*x_noe + reg.intercept_


# Shape
dos = ds['dos']
assert dos.min() == 0

tot_min_e = np.inf
tot_max_e = -np.inf

#f_vals = np.arange(energies.min(), energies.max(), 0.1)
f_vals = np.arange(0, 400, 0.1)

# Density of states for each volume, k_o, energy value
#    shape: (n_pq, n_ko, n_fvals)
f_dos = np.zeros((dos.shape[0], dos.shape[1], f_vals.size))


for i_pq, (pq,p,q) in enumerate(feat_pq):
    assert p == vals_pq[i_pq,0]
    assert q == vals_pq[i_pq,1]

    print('doing: p={}  q={}'.format(p,q))
    #this_fnot = f0[i_pq]

    for i_ko in range(pq+1):
        # Shape: (n_ko, n_noo, n_noe)
        this_dos = dos[i_pq, i_ko]

        occ = this_dos > 0
        # Shape same as this_dos
        this_f = energies[i_ko]

        occ_f = this_f[occ]
        occ_dos = this_dos[occ]

        assert np.isclose(occ_dos.sum(), binom(pq, i_ko))

        #f_all[i_pq, i_ko] = this_f

        # The bin indices for the energies
        f_bin_assign = np.digitize(occ_f, f_vals) - 1

        for mult, f_assign_idx in zip(occ_dos, f_bin_assign):
            f_dos[i_pq, i_ko, f_assign_idx] += mult

#get average f, as well as min,max, with ko
f_dos = f_dos[0]
max_f = np.zeros(f_dos.shape[0])
min_f = np.zeros_like(max_f)
mean_f = np.zeros_like(max_f)
ener_hist = np.zeros((f_dos.shape[0], f_vals.size))

for i_ko in range(f_dos.shape[0]):
    this_dos = f_dos[i_ko]

    occ = this_dos > 0

    max_f[i_ko] = f_vals[occ].max()
    min_f[i_ko] = f_vals[occ].min()

    ener_hist[i_ko] = np.log(this_dos)

np.savez_compressed("sam_dos_f_min_max", ko=np.arange(37), f_vals=f_vals, max_f=max_f, min_f=min_f, ener_hist=ener_hist)

