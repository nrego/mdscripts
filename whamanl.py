from __future__ import division, print_function

import numpy as np
from scipy.optimize import fmin_bfgs
from matplotlib import pyplot
import argparse
import logging
from mdtools import dr

from whamutils import gen_pdist

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
    parser.add_argument('--logweights', default='logweights.dat', 
                        help='Input file for WHAM logweights, if previously calculated')
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
        beta /= (args.T * 8.3144598e-3)

    for infile in args.input:
        log.info("Loading data {}".format(infile))
        ds = dr.loadPhi(infile)
        dat = np.array(ds.data[start:end]['$\~N$'])
        all_dat = np.append(all_dat, dat)
        n_samples.append(all_dat.size)

    n_samples = np.array(n_samples)

    bias_mat = np.zeros((all_dat.size, n_windows))
    for i, (ds_name, ds) in enumerate(dr.datasets.iteritems()):
        bias_mat[:,i] = beta * (((ds.kappa)/2.0) * (all_dat-ds.Nstar)**2 + (ds.phi*all_dat))
    
    embed()

    min_pt = 0
    max_pt = np.ceil(all_dat.max())+1

    binspace = 0.05
    binbounds = np.arange(0, np.ceil(max_pt)+binspace, binspace)
    bc = binbounds[:-1] + np.diff(binbounds)/2.0

    log.info("   ...Done")

    logweights = np.loadtxt(args.logweights)
    weights = np.exp(logweights) # for generating consensus distributions

    pdist = gen_pdist(all_dat, bias_mat, n_windows, logweights, binbounds)

    outarr = np.zeros((len(args.input), 3))

    for i, (ds_name, ds) in enumerate(dr.datasets.iteritems()):
        #if i > 0:
        #    embed()
        bias = beta*(0.5*ds.kappa*(bc-ds.Nstar)**2 + ds.phi*bc)

        # Consensus histogram
        consensus_pdist = np.exp(logweights[i] - bias) * pdist
        consensus_pdist = consensus_pdist / np.diff(binbounds)
        consensus_pdist = consensus_pdist / consensus_pdist.sum()

        # Observed (biased) histogram for this window i
        this_dat = np.array(ds.data[start:end]['$\~N$'])
        obs_hist, bb = np.histogram(this_dat, bins=binbounds)
        obs_hist = obs_hist / np.diff(binbounds)
        obs_hist = obs_hist / obs_hist.sum()

        comp_arr = np.zeros((bc.size, 3))
        comp_arr[:,0] = bc
        comp_arr[:,1] = obs_hist
        comp_arr[:,2] = consensus_pdist

        np.savetxt('nstar-{:05g}_phi-{:05g}_consensus.dat'.format(ds.Nstar, ds.phi*1000), comp_arr)

        shannonEntropy = np.nansum(obs_hist*np.log(obs_hist/consensus_pdist))
        log.info("  Shannon Entropy: {}".format(shannonEntropy))

        outarr[i, 0] = ds.phi
        outarr[i, 1] = ds.Nstar
        outarr[i, 2] = shannonEntropy

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