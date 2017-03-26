from __future__ import division, print_function

import numpy as np
from scipy.optimize import fmin_bfgs
from matplotlib import pyplot
import argparse
import logging
from mdtools import dr
import uwham
import matplotlib.pyplot as plt

import sys

import matplotlib as mpl
log = logging.getLogger()
log.addHandler(logging.StreamHandler())
Nfeval = 1

mpl.rcParams.update({'axes.labelsize': 30})
mpl.rcParams.update({'xtick.labelsize': 18})
mpl.rcParams.update({'ytick.labelsize': 18})
mpl.rcParams.update({'axes.titlesize': 36})


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
    parser.add_argument('--nbins', type=int, default=50,
                        help='Number of bins for each histogram (default 50)')
    parser.add_argument('--fmt', type=str, choices=['phi', 'xvg'], default='phi',
                        help='Format of input data files:  \'phi\' for phiout.dat;' \
                        '\'xvg\' for XVG type files (i.e. from alchemical GROMACS sims)')

    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.INFO)
    if args.debug:
        log.setLevel(logging.DEBUG)

    nbins = args.nbins

    log.info("Loading input files")

    if args.fmt == 'phi':
        for infile in args.input:
            dr.loadPhi(infile)
    elif args.fmt == 'xvg':
        for infile in args.input:
            dr.loadXVG(infile)

    log.info("   ...Done")

    start = args.start
    end = args.end 

    if args.fmt == 'xvg':
        for i, ds in enumerate(dr.datasets.values()):
            data = np.array(ds.data[start:end])

            lmbda = ds.lmbda

            if lmbda == 0:
                hist, bb = np.histogram(-data[:,-1], bins=nbins)

            else:
                # maybe try always comparing to lambda==1??
                hist, bb = np.histogram(data[:,0][np.abs(data[:,0]<1000.0)], bins=nbins)
                    
            bctrs = np.diff(bb)/2.0 + bb[:-1]
            plt.plot(bctrs, hist, label='$\lambda={}$'.format(lmbda))
            #embed()

        plt.legend()
        plt.show()