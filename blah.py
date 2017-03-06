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


phivals = np.arange(0.0, 10, 1)

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


thres_vals = np.linspace(0,1,500)
m = n_0.shape[0]
for i, phival in enumerate(phivals):
    curr_nvals = all_nvals[i]
    curr_data = np.zeros_like(thres_vals)
    for j, thres in enumerate(thres_vals):
        curr_data[j] = (curr_nvals <= thres).sum() / m
    plt.plot(thres_vals, curr_data, '-o', label=r'$\phi={}$'.format(phival))

thres_vals *= 5
for i, phival in enumerate(phivals):
    curr_rvals = all_rvals[i]
    if i == 0:
        curr_rvals /= curr_rvals
    curr_data = np.zeros_like(thres_vals)
    for j, thres in enumerate(thres_vals):
        curr_data[j] = (curr_rvals <= thres).sum() / m
    if i == 0:
        plt.step(thres_vals, curr_data, '-o', label=r'$\phi={}$'.format(phival))
    else:
        plt.plot(thres_vals, curr_data, '-o', label=r'$\phi={}$'.format(phival))

bins = np.linspace(0,0.1,50)
for i, phival in enumerate(phivals):
    curr_nvals = np.clip(all_nvals[i], 0, 0.1)
    if curr_nvals.max() >= bins.max():
        curr_bins = np.append(bins, curr_nvals.max()+0.01)
    else:
        curr_bins = bins.copy(0)
    hist, curr_bins = np.histogram(curr_nvals, normed=True, bins=curr_bins)

    plt.plot(curr_bins[:-1]+np.diff(curr_bins)/2.0, hist, label=r'$\phi={}$'.format(phival))

bins = np.linspace(0,2.1,51)
for i, phival in enumerate(phivals):
    if i == 0:
        continue
    curr_rvals = np.clip(all_rvals[i], 0.0, 2.0)

    curr_bins = bins.copy(0)
    hist, curr_bins = np.histogram(curr_rvals, normed=True, bins=75)

    plt.plot(curr_bins[:-1]+np.diff(curr_bins)/2.0, hist, label=r'$\phi={}$'.format(phival))
