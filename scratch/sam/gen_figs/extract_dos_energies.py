
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

#### RUN after 'extract_dos' (and after moving sam_dos to sam_data/data)
###
### This routine extracts the actual energies from the dos (since the dos is a fn of ko,noo,noe)
#
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
energies = reg.coef_[0]*x_ko + reg.coef_[1]*x_noo + reg.coef_[2]*x_noe #+ reg.intercept_


# Shape
dos = ds['dos']
assert dos.min() == 0

tot_min_e = np.inf
tot_max_e = -np.inf

#vals_f = np.arange(energies.min(), energies.max(), 0.1)
vals_f = np.arange(0, 500.1, 0.1)

# Density of states for each volume, k_o, energy value
#    shape: (n_pq, n_ko, n_fvals)
f_dos = np.zeros((dos.shape[0], dos.shape[1], vals_f.size))


for i_pq, (pq,p,q) in enumerate(feat_pq):
    assert p == vals_pq[i_pq,0]
    assert q == vals_pq[i_pq,1]

    print('doing: p={}  q={}'.format(p,q))
    #this_fnot = f0[i_pq]

    for i_ko in range(pq+1):
        # Shape: (n_noo, n_noe)
        this_dos = dos[i_pq, i_ko]

        occ = this_dos > 0
        # Shape same as this_dos
        this_f = energies[i_ko]

        occ_f = this_f[occ]
        assert occ_f.min() >= 0
        occ_dos = this_dos[occ]

        assert np.isclose(occ_dos.sum(), binom(pq, i_ko))

        #f_all[i_pq, i_ko] = this_f

        # The bin indices for the energies
        f_bin_assign = np.digitize(occ_f, vals_f) - 1

        for mult, f_assign_idx in zip(occ_dos, f_bin_assign):
            f_dos[i_pq, i_ko, f_assign_idx] += mult


## Get histograms of energies, noo's, and noe's with ko for each p,q
####################################################################

# Shape: (n_pq, n_ko, n_fvals)
ener_hist = np.zeros((vals_pq.shape[0], vals_ko.size, vals_f.size))
# Shape: (n_pq, n_ko, n_noo)
noo_hist = np.zeros((vals_pq.shape[0], vals_ko.size, vals_noo.size))
# Shape: (n_pq, n_ko, n_noe)
noe_hist = np.zeros((vals_pq.shape[0], vals_ko.size, vals_noe.size))

# Shape: (n_pq, n_ko)
max_f = np.zeros((vals_pq.shape[0], vals_ko.size))
min_f = np.zeros_like(max_f)
mean_f = np.zeros_like(max_f)

for i_pq in range(vals_pq.shape[0]):
    for i_ko in range(vals_ko.size):
        this_dos = dos[i_pq, i_ko]
        this_f_dos = f_dos[i_pq, i_ko]

        # Integrate over noe's
        this_noo_dos = this_dos.sum(axis=1)
        # Integrate over noo's
        this_noe_dos = this_dos.sum(axis=0)

        occ_f = this_f_dos > 0

        if occ_f.sum() > 0:
            max_f[i_pq, i_ko] = vals_f[occ_f].max()
            min_f[i_pq, i_ko] = vals_f[occ_f].min()
        # ko > pq
        else:
            max_f[i_pq, i_ko] = np.nan
            min_f[i_pq, i_ko] = np.nan

        ener_hist[i_pq, i_ko] = np.log(this_f_dos)
        noo_hist[i_pq, i_ko] = np.log(this_noo_dos)
        noe_hist[i_pq, i_ko] = np.log(this_noe_dos)



np.savez_compressed("data/sam_dos_f_min_max", vals_pq=vals_pq, vals_ko=vals_ko, vals_f=vals_f, vals_noo=vals_noo, vals_noe=vals_noe,
                    max_f=max_f, min_f=min_f, ener_hist=ener_hist, noo_hist=noo_hist, noe_hist=noe_hist)

