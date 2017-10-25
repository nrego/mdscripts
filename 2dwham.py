from __future__ import division, print_function

import os
import numpy as np
from whamutils import gen_U_nm, kappa, grad_kappa, gen_pdist
from scipy.optimize import fmin_l_bfgs_b as fmin_bfgs
from mdtools import dr
from matplotlib import cm
import matplotlib
import glob

dr.clearData()

# Each rstar umbrella
#dirs = ['rstar_{:03d}'.format(i) for i in [340,350,360,365,370,380,390,400,420,440]]
dirs = ['rstar_{:03d}'.format(i) for i in [4,5,6,7,8,9,10,11,12]]
dum_pos_z = 6
beta = 1 / (8.3144598e-3*300)

n_windows = 0
n_samples = []

all_dat_N = np.array([])
# distance of prot com from dummy
all_dat_r = np.array([])
# interface position - prot com
all_dat_rinter = np.array([])

# Get free energies of applying each rstar (w.r.t unbiased indus for each rstar, 'equil' dir)
logweights_rstar = np.loadtxt('logweights.dat')

n_rstar_windows = 0
# number of indus windows for each rstar window; array of ints: shape (n_rstar_windows, )
n_nstar_windows = []

# Total number of logweights - should be size n_windows at end
logweights = np.array([])
#logweights = np.loadtxt('logweights_all.dat')


# for each rstar umbrella...
for i, rootdirname in enumerate(dirs):
    n_rstar_windows += 1
    print("doing {}".format(rootdirname))
    os.chdir(rootdirname)
    #subdirs = !ls -d nstar*
    subdirs = sorted(glob.glob("nstar*"))

    # Will require an umbrella at this rstar with no nstar (INDUS) bias
    subdirs = list(np.append('equil', subdirs))

    this_logweights = np.loadtxt('logweights.dat') + logweights_rstar[i]
    logweights = np.append(logweights, this_logweights)
    # number of nstar biases in this rstar window
    n_nstar_windows.append(len(subdirs))

    os.chdir("..")

    # for each indus umbrella for this rstar
    for j, dirname in enumerate(subdirs):
        ds = dr.loadPhi("{}/{}/phiout.dat".format(rootdirname,dirname))
        dat_N = np.array(ds.data[1000:]['$\~N$'])
        all_dat_N = np.append(all_dat_N, dat_N)

        ds = dr.loadPmf("{}/{}".format(rootdirname,dirname))

        dat_r = np.array(ds.data[1000::5]['solDZ'])
        assert dat_r.ndim == dat_N.ndim == 1
        assert dat_r.size == dat_N.size
        all_dat_r = np.append(all_dat_r, dat_r)

        n_windows += 1
        n_samples.append(dat_r.size)

n_nstar_windows = np.array(n_nstar_windows)
n_samples = np.array(n_samples)
n_tot = n_samples.sum()
assert n_tot == all_dat_r.size == all_dat_N.size
assert n_rstar_windows == logweights_rstar.size
assert n_nstar_windows.sum() == n_samples.size

min_N = all_dat_N.min()
max_N = all_dat_N.max()
min_r = all_dat_r.min()
max_r = all_dat_r.max()
binbounds_N = np.arange(min_N, max_N+1, 1.0)
binbounds_r = np.arange(min_r, max_r+0.1, 0.005)
binctrs_N = np.diff(binbounds_N)/2.0 + binbounds_N[:-1]
binctrs_r = np.diff(binbounds_r)/2.0 + binbounds_r[:-1]

H, xedges, yedges = np.histogram2d(all_dat_N, all_dat_r, bins=(binbounds_N, binbounds_r))
H = H.T

X, Y = np.meshgrid(xedges, yedges)
Xc, Yc = np.meshgrid(binctrs_N, binctrs_r)

bias_mat = np.zeros((n_tot, n_windows), dtype=np.float32)

dsnames = dr.datasets.keys()

for i in xrange(n_windows):
    idx = i*2
    ds_N = dr.datasets[dsnames[idx]]
    ds_r = dr.datasets[dsnames[idx+1]]

    bias_mat[:, i] = beta*( (0.5*ds_N.kappa*(all_dat_N-ds_N.Nstar)**2) \
                         +(0.5*ds_r.kappa*(all_dat_r-ds_r.rstar)**2))

## For logweights
dr.clearData()

## accumulate the weights for each data point; shape: (n_tot, )

weights = np.zeros(n_tot)

for idx in xrange(n_tot):
    # the biases for this datapoint under each umbrella: shape (n_windows, )
    this_bias = bias_mat[idx, :]
    bias_factor = logweights - this_bias

    weights[idx] = 1.0/np.dot(n_samples, np.exp(bias_factor))

weights /= weights.sum()

binbounds_r_inter = dum_pos_z - binbounds_r
all_dat_r_inter = dum_pos_z - all_dat_r
binbounds_r_inter = binbounds_r_inter[::-1]

H_unbiased, xedges, yedges = np.histogram2d(all_dat_N, all_dat_r, bins=(binbounds_N, binbounds_r), weights=weights)
#H_unbiased = H_unbiased.T

loghist = -np.log(H_unbiased)
loghist -= loghist.min()

Xc, Yc = np.meshgrid(binctrs_r, binctrs_N)


matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('axes', labelsize=40)
matplotlib.rc('ytick', labelsize=30) 

plt.xlabel("$r$")
plt.ylabel("$N_v$")

cmap = cm.nipy_spectral
vmin = 0
vmax = 10
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
ax = plt.gca()
extent = (binbounds_r[0], binbounds_r[-1], binbounds_N[0], binbounds_N[-1])
im = ax.imshow(loghist, extent=extent, interpolation='nearest', origin='lower', alpha=0.75,
               cmap=cmap, norm=norm, aspect='auto')
cont = ax.contour(loghist, extent=extent, origin='lower', levels=np.arange(vmin,vmax,0.5),
                  colors='k', linewidths=1.0)

cb = plt.colorbar(im)
