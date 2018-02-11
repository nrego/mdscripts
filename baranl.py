from __future__ import division, print_function

import numpy
from matplotlib import pyplot
import argparse
import logging
from datareader import DataReader

from whamutils import gen_U_nm, kappa, grad_kappa, gen_pdist

import numpy as np
from scipy.optimize import fmin_l_bfgs_b as fmin_bfgs
import pymbar

from IPython import embed


fermi = lambda x : 1 / (1 + np.exp(x))

def iter_wham(dat0, dat1, c, n_sample0, n_sample1):

    sum0 = fermi(-dat0+c).sum()
    sum1 = fermi(dat1-c).sum()

    return np.log(sum0) - np.log(sum1) + c - np.log(n_sample1/n_sample0)

# Get the autocorrelation time for every pair of adjacent lambda windows (take highest)
# 
# Run BAR using all data
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Find free energy differences between adjacent lambdas using BAR")

    parser.add_argument('input', metavar='INPUT', type=str, nargs='+',
                        help='histogram.xvg input file')
    parser.add_argument('--debug', action='store_true',     
                        help='print debugging info')
    parser.add_argument('--autocorr-file', type=str, 
                        help='autocorrelation times input file (in ps)')
    parser.add_argument('--temp', '-T', type=float, default=300,
                        help='temperature in K (default: 300)')
    parser.add_argument('--start', '-b', default=0, type=float,
                        help='start time (in ps)')

    log = logging.getLogger()
    log.addHandler(logging.StreamHandler())

    args = parser.parse_args()

    start = args.start

    temp = args.temp
    beta = 1 / (0.0083144598*temp)

    if args.debug:
        log.setLevel(logging.DEBUG)

    infiles = args.input
    autocorr_file = args.autocorr_file

    dr = DataReader
    lmbdas = []

    for infile in infiles:
        print("Loading {}...".format(infile))
        ds = dr.loadXVG(infile)
        print("    ...done")

    datasets = dr.datasets.values()
    n_windows = len(datasets)

    autocorr = []

    for i in xrange(n_windows-1):
        ds0 = datasets[i]
        ds1 = datasets[i+1]

        lmbda_0 = ds0.lmbda
        lmbda_1 = ds1.lmbda

        dat0 = beta * np.array(ds0.data[2000:][lmbda_1])
        dat1 = beta *-np.array(ds1.data[2000:][lmbda_0])

        assert dat0.size == dat1.size

        n_samples = dat0.size

        iact0 = np.ceil(pymbar.timeseries.integratedAutocorrelationTime(dat0, fast=True))
        iact1 = np.ceil(pymbar.timeseries.integratedAutocorrelationTime(dat1, fast=True))

        print("lambda_0: {}".format(ds0.lmbda))
        print("    iact: {}".format(iact0))
        print("lambda_1: {}".format(ds1.lmbda))
        print("    iact: {}".format(iact1))

        print(" ")

        iact = max(iact0, iact1)

        # Number of steps (*0.02 gives time in ps) to extract uncorrelated samples
        ac_block = 1+(2*iact)
        autocorr.append(ac_block)

        #uncorr_n_sample = size // ac_block

        indices = np.arange(n_samples, dtype=np.int)

    autocorr = np.array(autocorr).astype(int)

embed()