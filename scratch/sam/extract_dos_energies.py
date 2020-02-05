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

ds = np.load('sam_dos.npz')

reg = np.load('sam_reg_total.npy').item()

# p,q combos we're considering
vals_pq = ds['vals_pq']

feat_pq = np.hstack((vals_pq.prod(axis=1)[:,None], vals_pq))
vals_ko = ds['vals_ko']
vals_noo = ds['vals_noo']
vals_noe = ds['vals_noe']

# Minimum f=dg_bind
min_dg = np.floor(np.dot(feat_pq, reg.coef_[:3]).min() - 1)
x_ko, x_noo, x_noe = np.meshgrid(vals_ko, vals_noo, vals_noe, indexing='ij')

# shape: (n_ko, n_noo, n_noe)
delta_f = reg.coef_[3]*x_ko + reg.coef_[4]*x_noo + reg.coef_[5]*x_noe
# Shape: (n_pq)
f0 = np.dot(feat_pq, reg.coef_[:3])


# Shape
dos = ds['dos']
assert dos.min() == 0

tot_min_e = np.inf
tot_max_e = -np.inf

f_vals = np.arange(-250, 50, 1)

# Density of states for each volume, k_o, energy value
#    shape: (n_pq, n_ko, n_fvals)
f_dos = np.zeros((dos.shape[0], dos.shape[1], f_vals.size))

for i_pq, (pq,p,q) in enumerate(feat_pq):
    assert p == vals_pq[i_pq,0]
    assert q == vals_pq[i_pq,1]

    print('doing: p={}  q={}'.format(p,q))
    this_fnot = f0[i_pq]

    for i_ko in range(pq+1):
        # Shape: (n_ko, n_noo, n_noe)
        this_dos = dos[i_pq, i_ko]

        occ = this_dos > 0
        # Shape same as this_dos
        this_f = this_fnot + delta_f[i_ko]

        occ_f = this_f[occ]
        occ_dos = this_dos[occ]

        # The bin indices for the energies
        f_bin_assign = np.digitize(occ_f, f_vals) - 1

        for mult, f_assign_idx in zip(occ_dos, f_bin_assign):
            f_dos[i_pq, i_ko, f_assign_idx] += mult

    #min_e = this_f[occ].min()
    #max_e = this_f[occ].max()

    #if min_e < tot_min_e:
    #    tot_min_e = min_e
    #if max_e > tot_max_e:
    #    tot_max_e = max_e

    #print('p: {}  q: {}  min_e: {:.2f}  max_e: {:.2f}'.format(p,q,min_e,max_e))

new_ds = dict()

for k, v in ds.items():
    new_ds[k] = v

new_ds['dos_f'] = f_dos
new_ds['vals_f'] = f_vals

np.savez_compressed('sam_dos', **new_ds)

'''
## Test
import itertools

# Expected dos for f vals for 2x2 system
expt_f = np.zeros_like(f_vals)

for k_c in range(5):
    for idx in itertools.combinations(np.arange(4), k_c):
        idx = np.array(idx).astype(int)
        state = State(idx, 2, 2)

        this_e = f0[0] + reg.coef_[3]*state.k_o + reg.coef_[4]*state.n_oo + reg.coef_[5]*state.n_oe

        e_assign = np.digitize(this_e, f_vals) - 1
        expt_f[e_assign] += 1
'''
