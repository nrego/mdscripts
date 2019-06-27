from __future__ import division, print_function

import numpy as np
from scipy.optimize import fmin_bfgs
from matplotlib import pyplot
import argparse
import logging
from mdtools import dr

from whamutils import gen_pdist, gen_data_logweights

from constants import k

import sys

import matplotlib as mpl

from IPython import embed

log = logging.getLogger()
log.addHandler(logging.StreamHandler())
Nfeval = 1

mpl.rcParams.update({'axes.labelsize': 30})
mpl.rcParams.update({'xtick.labelsize': 18})
mpl.rcParams.update({'ytick.labelsize': 18})
mpl.rcParams.update({'axes.titlesize': 36})


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run WHAM analysis (for INDUS datasets only!)")

    parser.add_argument('input', metavar='INPUT', type=str, nargs='+',
                        help='Input file names (presumably in a sensible order)')
    parser.add_argument('-b', '--start', type=int, default=0,
                        help='first timepoint (in ps)')
    parser.add_argument('-e', '--end', type=int, default=None,
                        help='last timepoint (in ps) - default is last available time point')
    parser.add_argument('--f_k', default='f_k.dat', 
                        help='Input file for WHAM f_k, if previously calculated')
    parser.add_argument('--autocorr', type=str, 
                        help='If provided, load list of autocorrelation times, in ps (default: none; use all points per window)')
    parser.add_argument('--logpdist', default='neglogpdist.dat', 
                        help='Name of calculated - log pdist file (default: neglogpdist.dat)')
    parser.add_argument('--debug', action='store_true',
                        help='print debugging info')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='be verbose')
    parser.add_argument('-T', metavar='TEMP', type=float,
                        help='convert Phi values to kT, for TEMP (K)')
    parser.add_argument('--plotPdist', action='store_true',
                        help='Plot resulting probability distribution')
    parser.add_argument('--plotE', action='store_true',
                        help='Plot resulting (free) energy distribution (-log(P))')
    parser.add_argument('--plotLogP', action='store_true',
                        help='Plot resulting log probability (log(P))')

    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.INFO)
    if args.debug:
        log.setLevel(logging.DEBUG)

    log.info("Loading input files")

    start = args.start
    end = args.end 

    n_windows = len(args.input)
    all_dat = np.array([])
    n_samples = []

    beta = 1
    if args.T:
        beta /= (args.T * k)

    ts = None
    for infile in args.input:
        log.info("Loading data {}".format(infile))
        ds = dr.loadPhi(infile)
        if ts is None:
            ts = ds.ts
        else:
            assert ts == ds.ts
        dat = np.array(ds.data[start:end]['$\~N$'])
        all_dat = np.append(all_dat, dat)
        n_samples.append(dat.size)

    if args.autocorr is not None:
        autocorr_nsteps = (1 + 2*(np.loadtxt(args.autocorr) / ts)).astype(int)
    else:
        autocorr_nsteps = np.ones(n_windows, dtype=int)
    
    n_samples = np.array(n_samples)

    bias_mat = np.zeros((all_dat.size, n_windows))
    for i, (ds_name, ds) in enumerate(dr.datasets.items()):
        bias_mat[:,i] = beta * (((ds.kappa)/2.0) * (all_dat-ds.Nstar)**2 + (ds.phi*all_dat))


    min_pt = 0
    max_pt = np.ceil(all_dat.max())

    binspace = 1
    binbounds = np.arange(0, max_pt+binspace, binspace)
    bc = binbounds[:-1] + np.diff(binbounds)/2.0

    log.info("   ...Done")

    f_k = np.loadtxt(args.f_k)

    logweights = gen_data_logweights(bias_mat, f_k, n_samples)

    outarr = np.zeros((len(args.input), 3))
    for i, (ds_name, ds) in enumerate(dr.datasets.items()):

        bias = -beta*(0.5*ds.kappa*(all_dat-ds.Nstar)**2 + ds.phi*all_dat)
        this_logweights = logweights+bias
        this_logweights -= this_logweights.max()
        this_weights = np.exp(this_logweights)
        this_weights /= this_weights.sum()

        # Consensus histogram
        consensus_pdist, bb = np.histogram(all_dat, bins=binbounds, weights=this_weights, normed=True)

        # Observed (biased) histogram for this window i
        this_dat = np.array(ds.data[start:end]['$\~N$'])
        obs_hist, bb = np.histogram(this_dat, bins=binbounds, normed=True)

        comp_arr = np.zeros((bc.size, 3))
        comp_arr[:,0] = bc
        comp_arr[:,1] = obs_hist
        comp_arr[:,2] = consensus_pdist

        #np.savetxt('nstar-{:05g}_phi-{:05g}_consensus.dat'.format(ds.Nstar, ds.phi*1000), comp_arr)

        kl_entropy = np.trapz(obs_hist*np.ma.log(obs_hist/consensus_pdist), bc)
        log.info("  Kullback-Leibler Relative Entropy: {}".format(kl_entropy))

        outarr[i, 0] = ds.phi
        outarr[i, 1] = ds.Nstar
        outarr[i, 2] = kl_entropy

    np.savetxt('wham_anl.dat', outarr, fmt='%1.2f %1.2f %1.4e')

    if args.plotPdist:
        pyplot.plot(probDist[:,0], probDist[:,1])
        pyplot.show()

    elif args.plotE:
        pyplot.plot(probDist[:,0], -np.log(probDist[:,1]))
        pyplot.show()

    elif args.plotLogP:
        pyplot.plot(probDist[:,0], np.log(probDist[:,1]))
        pyplot.show()