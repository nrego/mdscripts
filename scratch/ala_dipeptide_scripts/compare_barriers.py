from __future__ import division, print_function

import numpy as np
import matplotlib
mpl = matplotlib
from matplotlib import pyplot as plt
from matplotlib.image import NonUniformImage, imread
from scipy.optimize import minimize
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter

#import visvis as vv

#from IPython import embed

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from mdtools import dr

from constants import k

import os
import glob

from mdtools import dr

beta = 1/(k*300)


fnames = sorted(glob.glob('mu_*/boot_fn_payload.dat.npy'))


binbounds = np.arange(-180,184,4)
bc = np.diff(binbounds)/2.0 + binbounds[:-1]

alphas = []
avgs = []
all_data_avgs = []
for fname in fnames:

    dirname = os.path.dirname(fname)
    
    print("doing {}".format(dirname))

    alpha = float(dirname.split('_')[-1])
    alphas.append(alpha/10.0)

    payload = np.load(fname)
    n_boot_sample = payload.shape[0]
    loghists = []

    
    for i in range(n_boot_sample):
        weights, boot_data = payload[i]

        # Get everything with phi < 0
        mask = boot_data[:,0] < 0
        hist, bb = np.histogram(boot_data[mask,1], bins=binbounds, weights=weights[mask])

        loghist = -np.log(hist)
        loghist -= loghist.min()

        loghists.append(loghist)

    loghists = np.array(loghists)

    avg = loghists.mean(axis=0)
    errs = loghists.std(axis=0, ddof=1)

    #plt.errorbar(bc, avg, yerr=errs, label=r'$\beta \alpha={:0.2f}$'.format(alpha))
    out = np.dstack((bc, avg, errs)).squeeze()
    np.savetxt('{}/barrier_psi.dat'.format(dirname), out)

    avgs.append(avg)



    # From using all data (no error bars)
    payload_arr = np.load('{}/data_arr.npz'.format(dirname))['arr_0']

    phi_vals = payload_arr[:,0]
    psi_vals = payload_arr[:,1]
    ntwid_dat = payload_arr[:,2]
    nreg_dat = payload_arr[:,3]
    weights = payload_arr[:,4]

    mask = phi_vals < 0
    hist, bb = np.histogram(psi_vals[mask], bins=binbounds, weights=weights[mask])

    loghist = -np.log(hist)
    loghist -= loghist.min()
    all_data_avgs.append(loghist)

alphas = np.array(alphas) / 10.0




indices = [0,4,7,8,12]
barrier_slice = slice(42, 86, None)

for idx in indices:
    plt.errorbar(bc, avgs[idx], yerr=errs[idx], label=r'${}$'.format(alphas[idx]))

plt.legend()
