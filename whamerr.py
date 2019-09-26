from __future__ import division, print_function

import numpy as np
import pandas as pd
from matplotlib import pyplot
import argparse
import logging
from mdtools import dr, get_object
import scipy.integrate
#from scipy.optimize import fmin_l_bfgs_b as fmin_bfgs
from scipy.optimize import minimize
import pymbar
import time

from mdtools import ParallelTool

from constants import k

from whamutils import kappa, grad_kappa, gen_data_logweights, WHAMDataExtractor, callbackF

import matplotlib as mpl

import matplotlib.pyplot as plt


log = logging.getLogger('mdtools.whamerr')

from IPython import embed

def _subsample(bias_mat, uncorr_n_samples, uncorr_n_tot, n_samples, n_windows):

    # sub sample the bias matrix according to the number of uncorrelated samples from each window
    uncorr_bias_mat = np.zeros((uncorr_n_tot, n_windows), dtype=np.float64)
    start_idx = 0
    uncorr_start_idx = 0
    subsampled_indices = np.array([], dtype=int)

    np.random.seed()

    for i, this_uncorr_n_sample in enumerate(uncorr_n_samples):
        # the total number of (correlated) datapoints for this window
        this_n_sample = n_samples[i]
        avail_indices = np.arange(this_n_sample)
        # subsampled indices for this data
        this_indices = start_idx + np.random.choice(avail_indices, size=this_uncorr_n_sample, replace=True)
        subsampled_indices = np.append(subsampled_indices, this_indices)

        uncorr_bias_mat[uncorr_start_idx:uncorr_start_idx+this_uncorr_n_sample, :] = bias_mat[this_indices, :]
        
        uncorr_start_idx += this_uncorr_n_sample
        start_idx += this_n_sample


    return (uncorr_bias_mat, subsampled_indices)



def _bootstrap(lb, ub, uncorr_ones_m, uncorr_ones_n, bias_mat, n_samples, uncorr_n_samples,
               uncorr_n_sample_diag, uncorr_n_tot, n_windows, xweights, all_data, all_data_aux, boot_fn=None):

    # Number of bootstrap runs to do this round
    batch_size = ub - lb

    # Results for this bootstrap batch
    f_k_ret = np.zeros((batch_size, n_windows), dtype=np.float64)
    boot_fn_ret = np.zeros(batch_size, dtype=object)

    for batch_num in range(batch_size):

        ## Fill up the uncorrelated bias matrix
        boot_uncorr_bias_mat, boot_indices = _subsample(bias_mat, uncorr_n_samples, uncorr_n_tot, n_samples, n_windows)

        myargs = (boot_uncorr_bias_mat, uncorr_n_sample_diag, uncorr_ones_m, uncorr_ones_n, uncorr_n_tot)
        boot_f_k = np.append(0, minimize(kappa, xweights, method='L-BFGS-B', jac=grad_kappa, args=myargs).x)

        f_k_ret[batch_num,:] = boot_f_k
        
        if boot_fn is not None:
            boot_logweights = gen_data_logweights(boot_uncorr_bias_mat, boot_f_k, uncorr_n_samples, uncorr_ones_m, uncorr_ones_n)
            boot_fn_ret[batch_num] = boot_fn(all_data, all_data_aux, boot_indices, boot_logweights)
            del boot_logweights

        del boot_uncorr_bias_mat

    return (f_k_ret, boot_fn_ret, lb, ub)


class WHAMmer(ParallelTool):
    prog='WHAM/MBAR analysis'
    description = '''\
Perform MBAR/Binless WHAM analysis on 'phiout.dat' or '*.xvg' datasets (e.g. from alchemical FE cals with GROMACS).
Note that XVG type datasets must contain DeltaU for *every* other window (not just the adjacent window(s), as
    is required by 'g_bar', which uses TI, not MBAR).

Also perform bootstrapping standard error analysis - must specify an autocorrelation time for this to work correctly!

This tool supports parallelization (see options below)


-----------------------------------------------------------------------------
Command-line options
-----------------------------------------------------------------------------
'''
    
    def __init__(self):
        super(WHAMmer,self).__init__()
        
        # Parallel processing by default (this is not actually necessary, but it is
        # informative!)
        self.wm_env.default_work_manager = self.wm_env.default_parallel_work_manager

        self.beta = 1

        self.n_bootstrap = None

        self.output_filename = None

        self.start_weights = None

        self.boot_fn = None
    
        self.data_extractor = None

    # Total number of samples - sum of n_samples from each window
    @property
    def n_tot(self):
        return self.n_samples.sum()
    
    
    def add_args(self, parser):
        
        sgroup = parser.add_argument_group('(Binless) WHAM/MBAR error options')
        sgroup.add_argument('input', metavar='INPUT', type=str, nargs='+',
                            help='Input file names')
        sgroup.add_argument('--fmt', type=str, choices=['phi', 'pmf', 'simple'], default='phi',
                            help='Format of input data files:  \'phi\' for phiout.dat; \ '
                            '\'xvg\' for XVG type files (i.e. from alchemical GROMACS sims); \ '
                            '\'rama\' for RAMA type files (alanine dipeptide)')
        sgroup.add_argument('-b', '--start', type=int, default=0,
                            help='first timepoint (in ps) - default is first available time point')  
        sgroup.add_argument('-e', '--end', type=int, default=None,
                            help='last timepoint (in ps) - default is last available time point')
        sgroup.add_argument('--skip', type=int, default=None,
                            help='Skip this many picoseconds from each dataset. --autocorr option WILL BE IGNORED if this option is used')
        sgroup.add_argument('-T', metavar='TEMP', type=float,
                            help='convert Phi values to kT, for TEMP (K)')
        sgroup.add_argument('--bootstrap', type=int, default=1000,
                            help='Number of bootstrap samples to perform')   
        sgroup.add_argument('--autocorr', '-ac', type=float, help='Autocorrelation time (in ps); this can be \ '
                            'a single float, or one for each window') 
        sgroup.add_argument('--autocorr-file', '-af', type=str, 
                            help='Name of autocorr file (with times in ps for each window), if previously calculated')
        sgroup.add_argument('--nbins', type=int, default=25, help='number of bins, if plotting prob dist (default 25)')
        sgroup.add_argument('--f_k', type=str, default=None,
                            help='(optional) previously calculated f_k file for INDUS simulations - \ '
                            'if \'phi\' format option also supplied, this will calculate the Pv(N) (and Ntwid). \ '
                            'For \'xvg\' formats, this will calculate the probability distribution of whatever \ '
                              'variable has been umbrella sampled')
        sgroup.add_argument('--boot-fn', default=None, 
                            help='function, loaded from file of the form \'module.function\', to be performed \ '
                            'during each bootstrap iteration. If provided, the function is called during each bootstrap as: \ '
                            'fn(all_data, all_data_N, boot_indices, boot_logweights) where boot_indices corresponds to the indices\ '
                            'of this selected bootstrap sample, and boot_logweights are the corresponding (log of) statistical weights \ '
                            'for each bootstrap sample calculated with WHAM/MBAR.')

    ## Hackish way to 'inheret' attributes from the data extractor
    def __getattr__(self, attr):
        if self.data_extractor is not None:
            return getattr(self.data_extractor, attr)

    def process_args(self, args):

        self.beta = 1
        if args.T:
            self.beta /= (args.T * k)

        # Number of bootstrap samples to perform
        self.n_bootstrap = args.bootstrap

        if args.f_k:
            self.start_weights = np.loadtxt(args.f_k)
            log.info("starting weights: {}".format(self.start_weights))

        if args.autocorr_file is not None:
            auto = args.autocorr_file
        else:
            auto = args.autocorr


        log.info("Extracting data...")
        self.data_extractor = WHAMDataExtractor(args.input, args.fmt, auto, args.start, args.end, self.beta)
        log.info("Tau for each window: {} ps".format(self.autocorr))
        log.info("data time step: {} ps".format(self.ts))
        log.info("autocorr nsteps: {}".format(self.autocorr_blocks))        

        
        if args.boot_fn is not None:
            self.boot_fn = get_object(args.boot_fn)

    def go(self):


        if self.start_weights is not None:
            log.info("using initial weights: {}".format(self.start_weights))
            xweights = self.start_weights
        else:
            xweights = np.zeros(self.n_windows)

        assert xweights[0] == 0

        log.info("Quick sub-sampled MBAR run")
        uncorr_bias_mat, subsampled_indices = _subsample(self.bias_mat, self.uncorr_n_samples, self.uncorr_n_tot, self.n_samples, self.n_windows)
        
        myargs = (uncorr_bias_mat, self.uncorr_n_sample_diag, self.uncorr_ones_m, self.uncorr_ones_n, self.uncorr_n_tot)
        f_k_sub = minimize(kappa, xweights[1:], method='L-BFGS-B', args=myargs, jac=grad_kappa).x
        f_k_sub = np.append(0, f_k_sub)
        log.info("subsampled MBAR results: {}".format(f_k_sub))

        log.info("Running MBAR on entire dataset")
        log.info("...(this might take awhile)")
        myargs = (self.bias_mat, self.n_sample_diag, self.ones_m, self.ones_n, self.uncorr_n_tot)
        
        f_k_actual = minimize(kappa, f_k_sub[1:], method='L-BFGS-B', args=myargs, jac=grad_kappa, callback=callbackF).x
        f_k_actual = np.append(0, f_k_actual)
        log.info("MBAR results on entire dataset: {}".format(f_k_actual))

        np.savetxt('f_k_all.dat', f_k_actual, fmt='%3.6f')

        # Log of each datapoint's statistical weight. Note this accounts for statistical inefficiency in samples
        all_logweights = gen_data_logweights(self.bias_mat, f_k_actual, self.n_samples, self.ones_m, self.ones_n)
        
        # Now for bootstrapping...
        n_workers = self.work_manager.n_workers or 1
        batch_size = self.n_bootstrap // n_workers
        if self.n_bootstrap % n_workers != 0:
            batch_size += 1
        log.info("batch size for bootstrap: {}".format(batch_size))

        # the bootstrap estimates of free energies wrt window i=0
        f_k_boot = np.zeros((self.n_bootstrap, self.n_windows), dtype=np.float64)
        # Results of hook function, if desired
        boot_res = np.zeros(self.n_bootstrap, dtype=object)

        def task_gen():
            
            if __debug__:
                checkset = set()
            for lb in range(0, self.n_bootstrap, batch_size):
                ub = min(self.n_bootstrap, lb+batch_size)
          
                if __debug__:
                    checkset.update(set(range(lb,ub)))

                args = ()
                kwargs = dict(lb=lb, ub=ub, uncorr_ones_m=self.uncorr_ones_m, uncorr_ones_n=self.uncorr_ones_n, bias_mat=self.bias_mat,
                              n_samples=self.n_samples, uncorr_n_samples=self.uncorr_n_samples, uncorr_n_sample_diag=self.uncorr_n_sample_diag, 
                              uncorr_n_tot=self.uncorr_n_tot, n_windows=self.n_windows, xweights=f_k_actual[1:],
                              all_data=self.all_data, all_data_aux=self.all_data_aux, boot_fn=self.boot_fn)
                log.info("Sending job batch (from bootstrap sample {} to {})".format(lb, ub))
                yield (_bootstrap, args, kwargs)


        log.info("Beginning {} bootstrap iterations".format(self.n_bootstrap))
        # Splice together results into final array of densities
        for future in self.work_manager.submit_as_completed(task_gen(), queue_size=self.max_queue_len):
            f_k_slice, boot_res_slice, lb, ub = future.get_result(discard=True)
            log.info("Receiving result")
            f_k_boot[lb:ub, :] = f_k_slice
            log.debug("this boot weights: {}".format(f_k_slice))
            boot_res[lb:ub] = boot_res_slice
            del f_k_slice

        # Get SE from bootstrapped samples
        f_k_boot_mean = f_k_boot.mean(axis=0)
        f_k_se = np.sqrt(f_k_boot.var(axis=0))
        print('f_k (boot mean): {}'.format(f_k_boot_mean))
        print('f_k: {}'.format(f_k_actual))
        print('se: {}'.format(f_k_se))
        np.savetxt('err_f_k.dat', f_k_se, fmt='%3.6f')
        np.savetxt('boot_f_k.dat', f_k_boot)
        np.savetxt('f_k_all.dat', f_k_actual)
        np.savez_compressed('all_data.dat', logweights=all_logweights, data=self.all_data, data_aux=self.all_data_aux, bias_mat=self.bias_mat, n_samples=self.n_samples)


        if self.boot_fn is not None:
            np.save('boot_fn_payload.dat', boot_res)

if __name__=='__main__':
    WHAMmer().main()
