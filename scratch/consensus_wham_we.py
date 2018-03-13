from __future__ import division, print_function

import westpa
from fasthist import histnd, normhistnd
import numpy as np
import matplotlib
mpl = matplotlib
from matplotlib import pyplot as plt
from matplotlib.image import NonUniformImage, imread
from scipy.optimize import fmin_l_bfgs_b as fmin_bfgs
from whamutils import gen_U_nm, kappa, grad_kappa, gen_pdist
#import visvis as vv

from IPython import embed

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from mdtools import dr

import os
import glob


## For both of these functions, binbounds is 2d ([binbounds_phi, binbounds_psi])
def extract_we_data(dm, iter_start, binbounds):
    iter_stop = dm.current_iteration
    n_tot = dm.we_h5file['summary']['n_particles'][iter_start-1:iter_stop-1].sum()

    hist = np.zeros((binbounds[0].size-1, binbounds[1].size-1), dtype=np.float64)
    for i_iter in range(iter_start, iter_stop):
        iter_group = dm.get_iter_group(i_iter)

        phis = iter_group['pcoord'][:,-1,0]
        psis = iter_group['pcoord'][:,-1,1]
        ntwids = iter_group['auxdata/ntwid'][:,-1]
        nregs = iter_group['auxdata/nreg'][:,-1]
        weights = iter_group['seg_index']['weight']

        histnd(np.array([phis,psis]).T, binbounds, weights=weights, out=hist)

    hist = hist/hist.sum()

    return hist

def extract_wham_data(payload_arr, binbounds):
    phi_vals = payload_arr[:,0]
    psi_vals = payload_arr[:,1]
    weights = payload_arr[:,4]

    hist = histnd(np.array([phi_vals, psi_vals]).T, binbounds, weights=weights)

    hist = hist/hist.sum()

    return hist

binbounds = np.arange(-180,187,4)

we_iter_start = 1000


we_files = sorted(glob.glob('phi_*/west.h5'))

temp = 300
beta = 1/(temp*8.3144598e-3)
phi_vals = np.array([0, 2.0, 4.0, 5.0, 5.5, 6.0, 8.0, 10.0, 15.0, 20])

free_energies = np.array([  -0, 44.589521, 83.541158, 99.606209, 105.466612, 109.408471,116.394083, 118.821335, 120.544176, 121.066123])

wham_files = ['umbr/mu_{:03g}/data_arr.npz'.format(phi*10) for phi in phi_vals]

# Phi biases - Not to be confused with phi angles of pcoord
phi_vals = beta * phi_vals
n_windows = phi_vals.size

data_managers = []

wham_ents = []
we_ents = []
for i, we_filename in enumerate(we_files):
    dm = westpa.rc.new_data_manager()
    dm.we_h5filename = we_filename
    dm.open_backing()

    # phi_vals, psi_vals, ntwid, nreg, weights
    wham_payload = np.load(wham_files[i])['arr_0']
    
    we_hist = extract_we_data(dm, iter_start, [binbounds,binbounds])
    wham_hist = extract_wham_data(wham_payload, [binbounds, binbounds])
    del dm, wham_payload

    # Entropy w.r.t we_hist
    we_entropy = np.ma.fix_invalid(we_hist * np.log(we_hist/wham_hist)).sum()
    wham_entropy = np.ma.fix_invalid(wham_hist * np.log(wham_hist/we_hist)).sum()

    we_ents.append(we_entropy)
    wham_ents.append(wham_entropy)


