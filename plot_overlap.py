from __future__ import division, print_function

import numpy as np
from scipy.optimize import fmin_bfgs
from matplotlib import pyplot
import argparse
import logging
from mdtools import dr
import matplotlib.pyplot as plt
import pandas as pd

import sys

from whamutils import gen_pdist_xvg

import matplotlib as mpl
log = logging.getLogger()
log.addHandler(logging.StreamHandler())
Nfeval = 1

from IPython import embed

mpl.rcParams.update({'axes.labelsize': 30})
mpl.rcParams.update({'xtick.labelsize': 18})
mpl.rcParams.update({'ytick.labelsize': 18})
mpl.rcParams.update({'axes.titlesize': 36})

'''
A dirty, catch-all analysis script for assessing overlap in alchemical (lambda) 
    transformations (for use with MBAR/binless WHAM)

Analysis:

    *assume our dataset is an XVG of Delta U's for K windows (w/ lambda_i; i=0 to K-1)
    *Let Delta U_i = U_i+1 - U_i for all i = 0 to K-2
    *Analysis runs over pairs of windows (i, i+1), where i=0 to K-2
    *For each pair of windows (K-1 total pairs), we do:
        # plot P_i(DeltaU_i), P_{i+1}(Delta U_i)
        # calculate shannon entropy between each
        # plot log(P_i+1 (du)) - log(P_i (du)) + beta*du vs. du
'''


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run WHAM analysis")

    parser.add_argument('input', metavar='INPUT', type=str, nargs='+',
                        help='Input file names (presumably in a sensible order)')
    parser.add_argument('-b', '--start', type=int, default=0,
                        help='first timepoint (in ps)')
    parser.add_argument('-e', '--end', type=int, default=None,
                        help='last timepoint (in ps) - default is last available time point')
    parser.add_argument('--logweights', default='logweights.dat', 
                        help='Input file for WHAM logweights, if previously calculated')
    parser.add_argument('--debug', action='store_true',
                        help='print debugging info')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='be verbose')
    parser.add_argument('-T', metavar='TEMP', type=float,
                        help='convert Phi values to kT, for TEMP (K)')
    parser.add_argument('--skip', default=None, type=float,
                        help='only take data over this many time steps (in ps). Default: every data point')
    parser.add_argument('--binwidth', type=float, default=0.01,
                        help='Bin width over region of interest for delta U, in kT (default is 0.01 kT)')
    parser.add_argument('--overlap-cutoff', type=float, default=2.0,
                        help='Determines over which range to plot (and construct equal-spaced bins). Default: 2kT')
    parser.add_argument('--outname', default='{}_to_{}',
                        help='''base name for output plots. for every consecutive pair \
                        of windows i and i+1, an output file will be generated \
                        using python string formatting as follows: [outfile].format(i, i+1). \
                        The default is \'{}_{}\'; i.e. if i and i+1 are 0 and 1, the out \
                        file name is \'0_1\'.png''')
    parser.add_argument('--plotODM', action='store_true',
                        help='only plot ODMs if this option specified')


    args = parser.parse_args()
    outname = args.outname
    if args.verbose:
        log.setLevel(logging.INFO)
    if args.debug:
        log.setLevel(logging.DEBUG)


    log.info("Loading input files")

    logweights = np.loadtxt(args.logweights)
    logweight_diff = np.diff(logweights)

    d_g = np.diff(logweights)
    dsnames = []
    for infile in args.input:
        try:
            ds = dr.loadXVG(infile)
            dsnames.append(infile)
            log.info("loaded file {}".format(infile))
        except:
            log.error('error loading input {}'.format(infile))
            sys.exit()
    
    log.info("   ...Done")

    start = args.start
    end = args.end 
    dt = args.skip

    beta = 1/(8.3144598e-3 * args.T)

    n_windows = len(dsnames)

    bin_width = args.binwidth
    overlap_cutoff = args.overlap_cutoff

    entropies = np.zeros((n_windows-1, 4))
    #embed()
    if args.plotODM:
        for i in range(n_windows-1):
            ds_0 = dr.datasets[dsnames[i]]
            ds_1 = dr.datasets[dsnames[i+1]]

            lmbda_0 = ds_0.lmbda
            lmbda_1 = ds_1.lmbda

            entropies[i,0] = lmbda_0
            entropies[i,1] = lmbda_1

            if dt is not None:
                dt_0 = int(dt/ds_0.ts)
                dt_1 = int(dt/ds_1.ts)
            else:
                dt_0 = dt_1 = dt

            dat_0 = np.array(beta*ds_0.data[start:end:dt_0][lmbda_1])
            dat_1 = np.array(-beta*ds_1.data[start:end:dt_0][lmbda_0])

            min_floor = -100
            max_floor = 1000
            # Plot each distribution - initial plotting which might contain
            #   many wasted bins with little overlap
            abs_min_val = np.floor(min(dat_0.min(), dat_1.min()))
            abs_max_val = np.ceil(max(dat_0.max(), dat_1.max()))

            assert abs_min_val < abs_max_val
            log.info("Abs min delta U: {}; Abs max deta U: {}".format(abs_min_val, abs_max_val))

            #bb = np.linspace(abs_min_val, abs_max_val, 500)
            du = 0.05
            if abs_min_val < min_floor:
                bb = np.arange(min_floor, abs_max_val+du, du)
                bb = np.append(abs_min_val-du, bb)
            else:
                bb = np.arange(abs_min_val, abs_max_val+du, du)
            #embed()
            bc = np.diff(bb)/2.0 + bb[:-1]

            hist_0, blah = np.histogram(dat_0, bins=bb, normed=False)
            hist_1, blah = np.histogram(dat_1, bins=bb, normed=False)

            hist_0 = hist_0 / np.diff(bb)
            hist_1 = hist_1 / np.diff(bb)


            hist_0 /= np.sum(hist_0 * np.diff(bb))
            hist_1 /= np.sum(hist_1 * np.diff(bb))

            #embed()

            plt.title(r'$\lambda_0={}$ to $\lambda_1={}$'.format(lmbda_0, lmbda_1))
            plt.plot(bc, np.log(hist_0), label=r'$\ln{P_0(\Delta U)}$')
            plt.plot(bc, np.log(hist_1), label=r'$\ln{P_1(\Delta U)}$')
            plt.plot(bc, np.log(hist_1) - np.log(hist_0), label=r'$\ln{P_1(\Delta U)} - \ln{P_0(\Delta U)}$')

            odm = np.log(hist_1) - np.log(hist_0) + bc

            plt.xlabel(r'$\beta \Delta U$')
            plt.ylabel(r'$\ln{P(\Delta U)}$')
            plt.legend()
            plt.show()

            plt.title(r'$\lambda_0={}$ to $\lambda_1={}$'.format(lmbda_0, lmbda_1))
            plt.plot(bc, odm, '-o')

            plt.plot([bc[0], bc[-1]], [logweight_diff[i], logweight_diff[i]], label=r'\Delta A')

            plt.xlabel(r'$\beta \Delta U$')
            plt.ylabel(r'$\ln{P_1(\Delta U)} - \ln{P_0(\Delta U)} + \beta \Delta U$')

            plt.show()

    dudl = None
    bias_mat = None
    lmbdas = []
    datasets = []
    n_samples = []
    ## now get full pdist
    
    for i in range(n_windows):
        #embed()
        ds = dr.datasets[dsnames[i]]
        this_dudl = beta * np.array(ds.dhdl[start:end])
        datasets.append(this_dudl)
        n_samples.append(this_dudl.size)
        if dudl is None:
            dudl = this_dudl
        else:
            dudl = np.vstack((dudl, this_dudl))

        lmbdas.append(ds.lmbda)

    dudl = np.squeeze(dudl)
    lmbdas = np.array(lmbdas)
    
    binsize = 0.1

    themin = max(dudl.min(), -300)
    themax = min(dudl.max(), 100)
    binbounds = np.arange(themin, themax, binsize)
    bc = binbounds[:-1] + np.diff(binbounds)/2.0

    bias_mat = np.zeros((dudl.size, n_windows))

    start_idx = 0
    
    for i in range(n_windows):
        ds = dr.datasets[dsnames[i]]
        #bias_mat[:,i] = dudl * lmbdas[i]
        this_dat = np.array(ds.data[start:end][lmbdas])
        neg_u0 = this_dat[:,0]
        d_ui = this_dat - neg_u0[:, np.newaxis]

        bias_mat[start_idx:start_idx+n_samples[i], :] = beta * (d_ui)
        #bias_mat[start_idx:start_idx+n_samples[i], :] = beta * this_dat
        start_idx += n_samples[i]
    
    pdist = gen_pdist_xvg(dudl, bias_mat, n_samples, logweights, lmbdas, binbounds)

    pdist = pdist / np.diff(binbounds)
    pdist /= pdist.sum()
    
    entropies = np.zeros_like(lmbdas)

    for i in range(n_windows):
        this_lam = lmbdas[i]
        this_dat = datasets[i]

        this_hist, binbounds = np.histogram(this_dat, bins=binbounds)
        this_hist = this_hist / np.diff(binbounds)
        this_hist = this_hist / this_hist.sum()
        plt.plot(bc, this_hist, label=r'$\lambda={:.2f}$'.format(lmbdas[i]))
        this_weight = logweights[i]
        this_bias = this_lam * bc

        consensus_hist = np.exp(this_weight-this_bias) * pdist
        #consensus_hist /= consensus_hist.sum()

        entropy = np.nansum((this_hist * np.log(this_hist/consensus_hist)))
        entropies[i] = entropy

        log.info("  Shannon Entropy: {}".format(entropy))
    plt.legend()
    plt.show()
    loghist = -np.log(pdist)
    loghist -= loghist.min()
    embed()
    plt.plot(bc, loghist)
    plt.show()