from __future__ import division

import numpy as np
import os

n_0 = np.loadtxt('phi0.0/rho_avg.dat')
actual_data = np.loadtxt('out.dat')
actual_n_avg = actual_data[:,3]
actual_n_dep = actual_n_avg[0] - actual_n_avg

def get_data(phival):

    dirname = 'phi{:0.1f}/'.format(phival)


    rvals = np.loadtxt(os.path.join(dirname, 'rho_avg_norm.dat'))
    nvals = np.loadtxt(os.path.join(dirname, 'rho_avg.dat'))
    n_avg = nvals.sum()

    act_n_dep_curr = actual_n_dep[actual_data[:,0] == phival]
    act_n_avg_curr = actual_n_avg[actual_data[:,0] == phival]

    #estimate of total number of depleted waters
    n_dep = np.sum(n_0) - n_avg

    return (act_n_avg_curr, n_avg, act_n_dep_curr, n_dep, rvals, nvals)


phivals = np.arange(0.0, 6.5, 0.5)

all_n_avg = np.zeros_like(phivals)
all_act_n_avg = np.zeros_like(phivals)
all_n_dep = np.zeros_like(phivals)
all_act_n_dep = np.zeros_like(phivals)

all_rvals = np.zeros((phivals.shape[0], n_0.shape[0]))
all_nvals = np.zeros((phivals.shape[0], n_0.shape[0]))

for i,phival in enumerate(phivals):
    act_n_avg_curr, n_avg, act_n_dep_curr, n_dep, rvals, nvals = get_data(phival)

    all_n_avg[i] = n_avg
    all_act_n_avg[i] = act_n_avg_curr
    all_n_dep[i] = n_dep
    all_act_n_dep[i] = act_n_dep_curr
    all_rvals[i,...] = rvals
    all_nvals[i,...] = nvals

