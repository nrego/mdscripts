from __future__ import division, print_function

import westpa
from fasthist import histnd
import numpy as np


iter_start = 200

temp = 300
beta = 1/(temp*8.3144598e-3)
phi_vals = np.array([0, 5.5, 20])
free_energies = np.array([0.0, 105.7545, 121.1497])
files = ['phi_{:05g}/west.h5'.format(phi*1000) for phi in phi_vals]

binbounds = [np.arange(-181,182,1), np.arange(-181,182,1), np.arange(0.0, 80, 0.25)]
phi_binbounds = binbounds[0]
psi_binbounds = binbounds[1]
ntwid_binbounds = binbounds[2]

# Phi biases - Not to be confused with phi angles of pcoord
phi_vals *= beta
data_managers = []

for filename in files:
    dm = westpa.rc.new_data_manager()
    dm.we_h5filename = filename
    dm.open_backing()

    data_managers.append(dm)

# Total number of samples (n_iter * n_segs_per_iter) for each window
n_tots = np.zeros(phi_vals.size, dtype=int)

for i,dm in enumerate(data_managers):
    n_tot = dm.we_h5file['summary']['n_particles'][iter_start:dm.current_iteration-1].sum()

    n_tots[i] = n_tot


## Now for the meat...

# total histogram
hist = np.zeros((phi_binbounds.size-1, psi_binbounds.size-1, ntwid_binbounds.size-1), dtype=np.float64)

## For each window
for i, dm in enumerate(data_managers):

    # For each iteration
    for n_iter in xrange(iter_start, dm.current_iteration):
        iter_group = dm.get_iter_group(n_iter)

        phis = iter_group['pcoord'][:,-1,0]
        psis = iter_group['pcoord'][:,-1,1]
        ntwids = iter_group['auxdata/ntwid'][:,-1]
        weights = iter_group['seg_index']['weight']

        assert phis.shape[0] == psis.shape[0] == ntwids.shape[0] == weights.shape[0]

        n_segs = phis.shape[0]

        # stack the phi, psi, and ntwid vals for each seg into single array
        vals = np.array([phis,psis,ntwids]).T
        assert vals.shape == (n_segs, phi_vals.size)

        denom = np.dot(phi_vals[:,np.newaxis], ntwids[np.newaxis, :])
        assert denom.shape == (phi_vals.size, n_segs)

        denom -= free_energies[:, np.newaxis]
        denom = np.exp(-denom)
        denom *= n_tots[:, np.newaxis] 
        denom = np.sum(denom, axis=0)

        histnd(vals, binbounds, n_segs*weights/denom, out=hist, binbound_check=False)

# normalize the fucker
hist /= hist.sum()