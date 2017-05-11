from __future__ import division, print_function

import numpy as np
from scipy.optimize import fmin_bfgs
from matplotlib import pyplot
import argparse
import logging
from mdtools import dr
import matplotlib.pyplot as plt

import sys

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
    parser.add_argument('--dt', default=None, type=float,
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


    args = parser.parse_args()
    outname = args.outname
    if args.verbose:
        log.setLevel(logging.INFO)
    if args.debug:
        log.setLevel(logging.DEBUG)


    log.info("Loading input files")

    logweights = np.loadtxt(args.logweights)

    d_g = np.diff(logweights)

    for infile in args.input:
        try:
            dr.loadXVG(infile)
            log.info("loaded file {}".format(infile))
        except:
            log.error('error loading input {}'.format(infile))
            sys.exit()

    log.info("   ...Done")

    start = args.start
    end = args.end 
    dt = args.dt

    beta = 1/(8.3144598e-3 * args.T)

    dsnames = dr.datasets.keys()
    n_windows = len(dsnames)

    bin_width = args.binwidth
    overlap_cutoff = args.overlap_cutoff

    entropies = np.zeros((n_windows-1, 4))
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

        # Plot each distribution - initial plotting which might contain
        #   many wasted bins with little overlap
        abs_min_val = np.floor(min(dat_0.min(), dat_1.min()))
        abs_max_val = np.ceil(max(dat_0.max(), dat_1.max()))

        assert abs_min_val < abs_max_val
        log.info("Abs min delta U: {}; Abs max deta U: {}".format(abs_min_val, abs_max_val))

        bb2 = np.linspace(abs_min_val, abs_max_val, 500)
        bc2 = np.diff(bb2)/2.0 + bb2[:-1]
        hist_02, blah = np.histogram(dat_0, bins=bb2, normed=True)
        hist_12, blah = np.histogram(dat_1, bins=bb2, normed=True)

        # Find out data range with at least minimum overlap
        hist_diff = np.abs(np.log(hist_12)-np.log(hist_02))
        hist_diff[np.isnan(hist_diff) | np.isinf(hist_diff)] = np.float('inf')
        #embed()
        indices = np.where(hist_diff < overlap_cutoff)[0]
        min_index = indices.min()
        max_index = indices.max()
        min_val = np.floor(bb2[min_index])
        max_val = np.ceil(bb2[max_index])

        assert min_val < max_val

        log.info("Min delta U: {}; Max deta U: {}".format(min_val, max_val))

        # Now remake histograms
        #    Equal width bins are only assured over the region with 
        #    High overlap. large 'catch-all' bins are added to 
        #    capture datapoints outside this range. We do this, 
        #    rather than make all bins equal over the entire range
        #     in order to avoid possible rounding errors with np.arange 
        #       over an unnecessarily large range
        bb = np.arange(min_val, max_val, bin_width)
        if abs_min_val < min_val:
            bb = np.append(abs_min_val, bb)
        if abs_max_val > max_val:
            bb = np.append(bb, abs_max_val)

        hist_0, blah = np.histogram(dat_0, bins=bb)
        hist_1, blah = np.histogram(dat_1, bins=bb)
        #embed()
        min_count = 200

        min_count_idx = max(np.argmax(hist_0[1:] > min_count), np.argmax(hist_1[1:] > min_count)) + 1
        max_count_idx = max(np.argmax(hist_0[-1::-1] > min_count), np.argmax(hist_1[-1::-1] > min_count)) + 1
        max_count_idx = len(hist_0) - max_count_idx - 1
        hist_0 = hist_0 / np.diff(bb)
        hist_1 = hist_1 / np.diff(bb)

        norm_fac = np.sum(hist_0 * np.diff(bb))

        norm_diff = np.abs(norm_fac - np.sum(hist_1 * np.diff(bb)))
        if norm_diff > 1e-5:
            log.warning("  WARNING: difference in norm factors of {}".format(norm_diff))
        hist_0 = hist_0 / norm_fac
        hist_1 = hist_1 / norm_fac

        bc = np.diff(bb)/2.0 + bb[:-1]

        mask = (hist_0 > 0) & (hist_1 > 0)
        # Get shannon entropies
        int_0 = hist_0 * np.log(hist_0/hist_1)
        s_0 = np.trapz(int_0[mask], bc[mask])
        int_1 = hist_1 * np.log(hist_1/hist_0)
        s_1 = np.trapz(int_1[mask], bc[mask])

        entropies[i, 2] = s_0
        entropies[i, 3] = s_1

        # Find the 'overlapping distribution method' (odm)
        odm = np.log(hist_1) - np.log(hist_0) + bc

        plt.clf()
        plt.plot(bc, np.log(hist_0), label=r'$\ln{P_{0} (\Delta U)}$')
        plt.plot(bc, np.log(hist_1), label=r'$\ln{P_{1} (\Delta U)}$')
        plt.plot(bc, np.log(hist_1) - np.log(hist_0), label=r'$\ln{P_{1} (\Delta U)}-\ln{P_{0} (\Delta U)}$')
        
        plt.title('$\lambda_{}={}$ to $\lambda_{}={}$'.format(0, lmbda_0, 1, lmbda_1))
        plt.xlim(min_val, max_val)
        plt.ylim(-6,3)
        plt.xlabel(r'$\beta \Delta U$')
        plt.ylabel(r'$\ln{P(\Delta U)} \; (k_B T)$')
        plt.legend()
        out = outname + '_hist.png'
        plt.savefig(out.format(i, i+1), bbox_inches='tight')

        plt.clf()
        plt.plot(bc, odm, '-o')
        plt.plot([bc[0], bc[-1]], [d_g[i], d_g[i]], label=r'$\Delta G={}$'.format(d_g[i]))
        plt.xlim(bc[min_count_idx], bc[max_count_idx])
        plt.ylim(d_g[i]-0.2, d_g[i]+0.2)
        plt.xlabel(r'$\beta \Delta U$')
        plt.ylabel(r'$\ln{P_{1}(\Delta U)} - \ln{P_{0}(\Delta U)} + \beta \Delta U$ $(k_B T)$')
        plt.title('$\lambda_{}={}$ to $\lambda_{}={}$'.format(0, lmbda_0, 1, lmbda_1))
        plt.legend()
        out = outname + '_odm.png'
        #plt.show()
        plt.savefig(out.format(i, i+1), bbox_inches='tight')

        # Find the 'overlapping distribution method' (odm)
        odm2 = np.log(hist_12) - np.log(hist_02) + bc2

        plt.clf()
        plt.plot(bc2, np.log(hist_02), label=r'$\ln{P_{0} (\Delta U)}$')
        plt.plot(bc2, np.log(hist_12), label=r'$\ln{P_{1} (\Delta U)}$')
        plt.plot(bc2, np.log(hist_12) - np.log(hist_02), label=r'$\ln{P_{1} (\Delta U)}-\ln{P_{0} (\Delta U)}$')
        
        plt.title('$\lambda_{}={}$ to $\lambda_{}={}$'.format(0, lmbda_0, 1, lmbda_1))
        plt.xlim(max(abs_min_val, -50), min(abs_max_val, 50))
        plt.xlabel(r'$\beta \Delta U$')
        plt.ylabel(r'$\ln{P(\Delta U)} \; (k_B T)$')
        plt.legend()
        out = outname + '_hist2.png'
        plt.savefig(out.format(i, i+1), bbox_inches='tight')

        plt.clf()
        plt.plot(bc2, odm2, '-o')
        plt.plot([bc2[0], bc2[-1]], [d_g[i], d_g[i]], label=r'$\Delta G={}$'.format(d_g[i]))
        plt.xlim(max(abs_min_val, -50), min(abs_max_val, 50))
        plt.ylim(d_g[i]-2, d_g[i]+2)
        plt.xlabel(r'$\beta \Delta U$')
        plt.ylabel(r'$\ln{P_{1}(\Delta U)} - \ln{P_{0}(\Delta U)} + \beta \Delta U$ $(k_B T)$')
        plt.title('$\lambda_{}={}$ to $\lambda_{}={}$'.format(0, lmbda_0, 1, lmbda_1))
        plt.legend()
        out = outname + '_odm2.png'
        #plt.show()
        plt.savefig(out.format(i, i+1), bbox_inches='tight')

    headerstr = 'lambda_0 lambda_1 s_0 s_1'
    np.savetxt('entropies.dat', entropies, fmt='%1.2f %1.2f %1.5f %1.5f', header=headerstr)
