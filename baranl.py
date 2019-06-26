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

from matplotlib import pyplot as plt

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
    logweights = []
    err_logweights = []

    n_boot = 128
    binwidth = 0.05

    for i in range(n_windows-1):
        ds0 = datasets[i]
        ds1 = datasets[i+1]

        lmbda_0 = ds0.lmbda
        lmbda_1 = ds1.lmbda

        dat0 = beta * np.array(ds0.data[2000:][lmbda_1])
        dat1 = beta *-np.array(ds1.data[2000:][lmbda_0])

        minval = min(dat0.min(), dat1.min())
        maxval = max(dat0.max(), dat1.max())

        binbounds = np.arange(minval, maxval, binwidth)
        bc = (binbounds[:-1] + binbounds[1:]) / 2.0

        assert dat0.size == dat1.size

        n_samples = dat0.size

        iact0 = pymbar.timeseries.integratedAutocorrelationTime(dat0, fast=True)
        iact1 = pymbar.timeseries.integratedAutocorrelationTime(dat1, fast=True)

        print("lambda_0: {}".format(ds0.lmbda))
        print("    iact: {}".format(iact0))
        print("lambda_1: {}".format(ds1.lmbda))
        print("    iact: {}".format(iact1))

        print(" ")


        # Number of steps (*0.02 gives time in ps) to extract uncorrelated samples
        ac_block0 = int(np.ceil(1+(2*iact0)))
        ac_block1 = int(np.ceil(1+(2*iact1)))
        #autocorr.append(ac_block)

        uncorr_n_sample = np.array([n_samples / ac_block0, n_samples / ac_block1]).astype(int)
        uncorr_n_sample += 1
        
        # Number of effective total samples
        uncorr_n_tot = uncorr_n_sample.sum()
        
        uncorr_bias_mat = np.zeros((uncorr_n_tot, 2))
        uncorr_bias_mat[:uncorr_n_sample[0], 1] = dat0[::ac_block0]
        uncorr_bias_mat[uncorr_n_sample[0]:, 1] = dat1[::ac_block1]
        
        uncorr_n_sample_diag = np.matrix(np.diag(uncorr_n_sample / uncorr_n_tot), dtype=np.float32)
        uncorr_ones_m = np.matrix(np.ones(2,), dtype=np.float32).T
        uncorr_ones_n = np.matrix(np.ones(uncorr_n_tot,), dtype=np.float32).T

        uncorr_myargs = (uncorr_bias_mat, uncorr_n_sample_diag, uncorr_ones_m, uncorr_ones_n, uncorr_n_tot)
        xweights = np.array([0.])

        hist0, bb = np.histogram(dat0[::ac_block0], bins=binbounds, normed=True)
        hist1, bb = np.histogram(dat1[::ac_block1], bins=binbounds, normed=True)

        boot_bias_mat = np.zeros_like(uncorr_bias_mat)
        ## Bootstrap 
        boot_wts = np.zeros((n_boot, 1))

        for i in range(n_boot):
            indices0 = np.random.choice(n_samples, uncorr_n_sample[0])
            indices1 = np.random.choice(n_samples, uncorr_n_sample[1])

            boot_bias_mat[:uncorr_n_sample[0], 1] = dat0[indices0]
            boot_bias_mat[uncorr_n_sample[0]:, 1] = dat1[indices1]
            
            boot_args = (boot_bias_mat, uncorr_n_sample_diag, uncorr_ones_m, uncorr_ones_n, uncorr_n_tot)
            res = fmin_bfgs(kappa, xweights, fprime=grad_kappa, args=boot_args)[0]

            boot_wts[i] = -res[0]

        logwt = boot_wts.mean()
        logweights.append(logwt)
        err_logweights.append(boot_wts.std(ddof=1))

        plt.plot(bc, np.log(hist1)-np.log(hist0)+bc, '-o')
        plt.plot([bc[0], bc[-1]], [logwt, logwt])
        plt.show()
    
    logweights = np.array(logweights)
    err_logweights = np.array(err_logweights)
    embed()
