from __future__ import division, print_function

import numpy as np
import pandas as pd
from matplotlib import pyplot
import argparse
import logging
from mdtools import dr
import scipy.integrate
from scipy.optimize import fmin_l_bfgs_b as fmin_bfgs
import pymbar
import time

from fasthist import normhistnd

from mdtools import ParallelTool

from whamutils import gen_U_nm, kappa, grad_kappa, gen_pdist

import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rcParams.update({'axes.labelsize': 50})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 40})
mpl.rcParams.update({'axes.titlesize': 50})
#mpl.rcParams.update({'titlesize': 42})

log = logging.getLogger('mdtools.whamerr')

from IPython import embed


## Perform bootstrapped MBAR/Binless WHAM analysis for phiout.dat or *.xvg datasets (e.g. from FE calcs in GROMACS)
#    Note for .xvg datasets (alchemical free energy calcs in gromacs), each file must contain *every* other window
#       (not just the adjacent windows, as g_bar requires for TI)


# Load list of infiles from start
# Return range over entire dataset (as tuple)
def load_range_data(infiles, start, end):
    minval = float('inf')
    maxval = float('-inf')
    for f in infiles:
        ds = dr.loadPhi(f)

        # Min, max for this dataset
        tmpmax = ds.max(start=start, end=end)['$\~N$']
        tmpmin = ds.min(start=start, end=end)['$\~N$']

        minval = tmpmin if minval>tmpmin else minval
        maxval = tmpmax if maxval<tmpmax else maxval


    return minval, maxval

   
def parse_np_array(inputarr):
    namespace = {'np': np,
                 'inf': float('inf')}
                 
    try:
        binspec_compiled = eval(inputarr,namespace)
    except Exception as e:
        raise ValueError('invalid bin specification: {!r}'.format(e))


# phidat is datareader (could have different number of values for each ds)
# phivals is (nphi, ) array of phi values (in kT)
def _bootstrap(lb, ub, ones_m, ones_n, bias_mat, n_samples, n_boot_samples,
               n_boot_sample_diag, n_boot_tot, n_windows, autocorr_nsteps, 
               xweights, binbounds, all_data_N):

    np.random.seed()
    batch_size = ub - lb
    logweights_ret = np.zeros((batch_size, n_windows), dtype=np.float32)
    if binbounds is not None and all_data_N is not None:
        neglogpdist_N_ret = np.zeros((batch_size, binbounds.shape[0]-1))
    else:
        neglogpdist_N_ret = None

    for batch_num in xrange(batch_size):

        # Will contain bootstrap samples from biasMat
        bias_mat_boot = np.zeros((n_boot_tot, n_windows), dtype=np.float32)
        # Subsample bias_mat
        i_start = 0
        i_boot_start = 0

        # We will get n_sample datapoints from window i for this bootstrap batch_num
        for i, n_sample in enumerate(n_samples):
            # the entire bias matrix for window i, shape (nsample, nwindows)
            bias_mat_window = bias_mat[i_start:i_start+n_sample, :]

            block_size = autocorr_nsteps[i]
            num_blocks = n_sample // block_size
            
            boot_n_sample = num_blocks * block_size

            # Slice of the bootstrap matrix for this window
            bias_mat_boot_window = bias_mat_boot[i_boot_start:i_boot_start+boot_n_sample, :]
            # Blargh - sanity check
            assert boot_n_sample == n_boot_samples[i]

            avail_start_indices = n_sample - block_size + 1
            assert avail_start_indices <= n_sample

            boot_start_indices = np.random.randint(avail_start_indices, size=num_blocks)
            
            #embed()
            # k runs from 0 to num_blocks for window i
            for k, boot_start_idx in enumerate(boot_start_indices):
                start_idx = k*block_size
                bias_mat_boot_window[start_idx:start_idx+block_size, :] = bias_mat_window[boot_start_idx:boot_start_idx+block_size, :]
            
            i_start += n_sample
            i_boot_start += boot_n_sample

        # WHAM this bootstrap sample

        myargs = (bias_mat_boot, n_boot_sample_diag, ones_m, ones_n, n_boot_tot)
        this_weights = -np.array(fmin_bfgs(kappa, xweights, fprime=grad_kappa, args=myargs)[0])
        logweights_ret[batch_num, 1:] = this_weights
        del bias_mat_boot, bias_mat_window, bias_mat_boot_window

        # Get -ln(Pv(N)) using WHAM results for this bootstrap sample
        if binbounds is not None and all_data_N is not None:
            this_pdist_N = gen_pdist(all_data_N, bias_mat, n_samples, logweights_ret[batch_num].astype(np.float64), binbounds)
            this_pdist_N /= (this_pdist_N * np.diff(binbounds)).sum()
            
            neglogpdist_N_ret[batch_num, :] = -np.log(this_pdist_N)

    return (logweights_ret, neglogpdist_N_ret, lb, ub)


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

        # 'phi' or 'xvg'
        self.fmt = None

        # Number of input simulations (windows) - arranged in order of input file
        self.n_windows = None

        # All data from all windows to be WHAMed, in 1-d array
        # shape: (n_tot,)
        self.all_data = None

        # Only initialized for 'phi' datasets - same shape as above
        self.all_data_N = None

        # bias for each window
        # shape: (n_tot, n_windows)
        #    Each row contains ith observation at *each* biasing window
        #    Each column corresponds to a single umbrella window
        self.bias_mat = None

        # (n_windows,) array of number of samples in each window
        self.n_samples = None

        self.nbins = None

        self.beta = 1

        self.n_bootstrap = None

        # Autocorr time for each window (in ps)
        self.autocorr = None
        self.ts = None

        self.dr = dr
        self.output_filename = None

        self.start_weights = None
        self.for_lmbdas = []
    
    # Total number of samples - sum of n_samples from each window
    @property
    def n_tot(self):
        return self.n_samples.sum()
    
    
    def add_args(self, parser):
        
        sgroup = parser.add_argument_group('(Binless) WHAM/MBAR error options')
        sgroup.add_argument('input', metavar='INPUT', type=str, nargs='+',
                            help='Input file names')
        sgroup.add_argument('--fmt', type=str, choices=['phi', 'xvg'], default='phi',
                            help='Format of input data files:  \'phi\' for phiout.dat; \ '
                            '\'xvg\' for XVG type files (i.e. from alchemical GROMACS sims)')
        sgroup.add_argument('-b', '--start', type=int, default=0,
                            help='first timepoint (in ps) - default is first available time point')  
        sgroup.add_argument('-e', '--end', type=int, default=None,
                            help='last timepoint (in ps) - default is last available time point')
        sgroup.add_argument('-T', metavar='TEMP', type=float,
                            help='betaert Phi values to kT, for TEMP (K)')
        sgroup.add_argument('--bootstrap', type=int, default=1000,
                            help='Number of bootstrap samples to perform')   
        sgroup.add_argument('--autocorr', '-ac', type=float, help='Autocorrelation time (in ps); this can be \ '
                            'a single float, or one for each window') 
        sgroup.add_argument('--autocorr-file', '-af', type=str, 
                            help='Name of autocorr file (with times in ps for each window), if previously calculated')
        sgroup.add_argument('--min-autocorr-time', type=float, default=0,
                            help='The minimum autocorrelation time to use (in ps). Default is no minimum')
        sgroup.add_argument('--nbins', type=int, default=25, help='number of bins, if plotting prob dist (default 25)')
        sgroup.add_argument('--logweights', type=str, default=None,
                            help='(optional) previously calculated logweights file for INDUS simulations - \ '
                            'if \'phi\' format option also supplied, this will calculate the Pv(N) (and Ntwid). \ '
                            'For \'xvg\' formats, this will calculate the probability distribution of whatever \ '
                              'variable has been umbrella sampled')


    def process_args(self, args):

        self.fmt = args.fmt
        self.n_windows = 0
        try:
            for i, infile in enumerate(args.input):
                log.info("loading {}th input: {}".format(i, infile))

                if self.fmt == 'phi': 
                    self.dr.loadPhi(infile) 
                elif self.fmt == 'xvg': 
                    ds = self.dr.loadXVG(infile)
                    self.for_lmbdas.append(ds.lmbda)
                self.n_windows += 1
        except:
            raise IOError("Error: Unable to successfully load inputs")
        #embed()
        self.for_lmbdas = pd.Index(sorted(self.for_lmbdas))
        self.min_autocorr_time = args.min_autocorr_time

        self.beta = 1
        if args.T:
            self.beta /= (args.T * 8.3144598e-3)

        # Number of bootstrap samples to perform
        self.n_bootstrap = args.bootstrap

        if args.logweights:
            self.start_weights = -np.loadtxt(args.logweights)
            log.info("starting weights: {}".format(self.start_weights))

        if args.autocorr_file:
            self._parse_autocorr_list(args.autocorr_file)
        if args.autocorr:
            self._parse_autocorr(args.autocorr)
        self.unpack_data(args.start, args.end)

        self.nbins = args.nbins
    
    def _parse_autocorr(self, autocorr):
        self.autocorr = np.ones(self.n_windows)
        if autocorr < self.min_autocorr_time:
            log.warning("Supplied autocorr time ({} ps) is less than minimum autocorr time ({} ps). Setting autocorr times to {} ps.".format(autocorr, self.min_autocorr_time, self.min_autocorr_time))
        self.autocorr *= max(autocorr, self.min_autocorr_time)

    def _parse_autocorr_list(self, autocorrfile):
        autocorr_times = np.loadtxt(autocorrfile)

        for i, autocorr in enumerate(autocorr_times):
            if autocorr < self.min_autocorr_time:
                log.warning("Supplied autocorr time ({} ps) is less than minimum autocorr time ({} ps). Setting autocorr times to {} ps.".format(autocorr, self.min_autocorr_time, self.min_autocorr_time))
                autocorr_times[i] = self.min_autocorr_time

        self.autocorr = autocorr_times

    # Note: sets self.ts, as well
    def unpack_data(self, start, end):
        if self.fmt == 'phi':
            self._unpack_phi_data(start, end)

        elif self.fmt == 'xvg':
            self._unpack_xvg_data(start, end)

    # Put all data points into N dim vector
    def _unpack_phi_data(self, start, end=None):

        self.all_data = np.array([], dtype=np.float32)
        self.all_data_N = np.array([]).astype(int)
        self.n_samples = np.array([]).astype(int)

        if self.autocorr is None:
            do_autocorr = True
            self.autocorr = np.zeros(self.n_windows)
        else:
            do_autocorr = False

        for i, (ds_name, ds) in enumerate(self.dr.datasets.iteritems()):

            if self.ts == None:
                self.ts = ds.ts
            # Sanity check - every input should have same timestep
            else:
                np.testing.assert_almost_equal(self.ts, ds.ts)

            data = ds.data[start:end]
            dataframe = np.array(data['$\~N$'])
            dataframe_N = np.array(data['N']).astype(np.int32)

            if do_autocorr:
                autocorr_len = np.ceil(pymbar.timeseries.integratedAutocorrelationTime(dataframe[:]))
                self.autocorr[i] = max(self.ts * autocorr_len, self.min_autocorr_time)

            self.n_samples = np.append(self.n_samples, dataframe.shape[0])

            self.all_data = np.append(self.all_data, dataframe)
            self.all_data_N = np.append(self.all_data_N, dataframe_N)

        self.bias_mat = np.zeros((self.n_tot, self.n_windows), dtype=np.float32)

        # Ugh !
        for i, (ds_name, ds) in enumerate(self.dr.datasets.iteritems()):
            self.bias_mat[:, i] = self.beta*(0.5*ds.kappa*(self.all_data-ds.Nstar)**2 + ds.phi*self.all_data) 

        dr.clearData()

    # Put all data points into N dim vector
    def _unpack_xvg_data(self, start, end=None):

        self.all_data = None
        self.n_samples = np.array([], dtype=np.int32)
        self.bias_mat = None

        if self.autocorr is None:
            do_autocorr = True
            self.autocorr = np.zeros(self.n_windows)
        else:
            do_autocorr = False

        for i, (ds_name, ds) in enumerate(self.dr.datasets.iteritems()):
            #embed()

            log.info("Unpacking {}th dataset ({:s})".format(i, ds_name))

            if self.ts == None:
                self.ts = ds.ts
            else:
                np.testing.assert_almost_equal(self.ts, ds.ts)

            ## quick test that dataset makes sense - DU's for its own lambda should be all zeros
            arr_self = ds.data[ds.lmbda]
            np.testing.assert_array_almost_equal(arr_self, np.zeros_like(arr_self), decimal=6)

            dataframe = np.array(ds.data[start:end][self.for_lmbdas], dtype=np.float32)

            if do_autocorr:
                log.info("    Calculating autocorrelation time...")
                autocorr_nsteps = 1
                n_samples = dataframe.shape[0]
                max_autocorr_len = n_samples // 50
                for k in [i-1, i+1]:
                    if k < 0:
                        continue
                    try:
                        autocorr_res = 2 * np.ceil(pymbar.timeseries.integratedAutocorrelationTime(dataframe[:,k]))
                        if autocorr_res > autocorr_nsteps:
                            autocorr_nsteps = autocorr_res
                    except:
                        continue
                self.autocorr[i] = max(self.ts * autocorr_nsteps * 0.5, self.min_autocorr_time)
                log.info("      Tau={} ps".format(self.autocorr[i]))
            bias = self.beta*dataframe # biased values for all windows
            self.n_samples = np.append(self.n_samples, dataframe.shape[0])

            np.savetxt('autocorr', self.autocorr)

            if self.all_data is None:
                self.all_data = dataframe
            else:
                self.all_data = np.vstack((self.all_data, dataframe))

            if self.bias_mat is None:
                self.bias_mat = bias
            else:
                self.bias_mat = np.vstack((self.bias_mat, bias))

        dr.clearData()
        
    def go(self):

        # 2*autocorr length (ps), divided by step size (also in ps) - the block_size(s)
        autocorr_nsteps = np.ceil(2*self.autocorr/self.ts).astype(int)
        log.info("autocorr length: {} ps".format(self.autocorr))
        log.info("data time step: {} ps".format(self.ts))
        log.info("autocorr nsteps: {}".format(autocorr_nsteps))
        # Number of complete blocks for each sample window - the effective 
        #    number of uncorrelated samples for each window
        n_blocks = self.n_samples // autocorr_nsteps

        # Find the number of **uncorrelated** samples for each window
        n_uncorr_samples = n_blocks #* autocorr_nsteps
        # Size of each bootstrap for each window
        n_uncorr_tot  = n_uncorr_samples.sum()
        assert n_uncorr_tot <= self.n_tot

        # Diagonal is fractional n_boot_samples for each window - should be 
        #    float because future's division function
        n_uncorr_sample_diag = n_boot_sample_diag = np.matrix( np.diag(n_uncorr_samples / n_uncorr_tot), dtype=np.float32 )

        n_sample_diag = np.matrix( np.diag(self.n_samples / self.n_tot), dtype=np.float32)
        # Run WHAM on entire dataset - use the logweights as inputs to future bootstrap runs
        # (k x 1) ones vector; k is number of windows
        ones_m = np.matrix(np.ones(self.n_windows,), dtype=np.float32).T
        # (n_tot x 1) ones vector; n_tot = sum(n_k) total number of samples over all windows
        ones_n = np.matrix(np.ones(self.n_tot,), dtype=np.float32).T

        # Do MBAR on *entire* dataset, but weight each window's contribution according
        #    to its number of *uncorrelated* samples
        myargs = (self.bias_mat, n_sample_diag, ones_m, ones_n, self.n_tot)
        
        if self.start_weights is not None:
            log.info("using initial weights: {}".format(self.start_weights))
            xweights = self.start_weights[1:]
        else:
            xweights = np.zeros(self.n_windows-1)

        log.info("Running MBAR on entire dataset")
        # fmin_bfgs spits out a tuple with some extra info, so we only take the first item (the weights)
        logweights_actual = fmin_bfgs(kappa, xweights, fprime=grad_kappa, args=myargs)[0]
        logweights_actual = -np.append(0, logweights_actual)
        log.info("MBAR results on entire dataset: {}".format(logweights_actual))

        ## TODO: this is messy - this if statement is only used for plotting/reweighing
        #       purposes when using INDUS datasets (which I misleadingly call 'phi' data)
        if self.fmt == 'phi':
            max_n = int(np.ceil(max(self.all_data_N.max(), self.all_data.max())))
            min_n = int(np.floor(min(self.all_data_N.min(), self.all_data.min())))
            log.info("Min: {:d}, Max: {:d}".format(min_n, max_n))
            binbounds = np.arange(min_n,max_n+2,1)
            neglogpdist_N_boot = np.zeros((self.n_bootstrap, binbounds.shape[0]-1), dtype=np.float64)
        else:
            binbounds = None

        np.savetxt('logweights.dat', logweights_actual, fmt='%3.6f')
        

        # Now for bootstrapping...
        n_workers = self.work_manager.n_workers or 1
        batch_size = self.n_bootstrap // n_workers
        if self.n_bootstrap % n_workers != 0:
            batch_size += 1
        log.info("batch size: {}".format(batch_size))

        logweights_boot = np.zeros((self.n_bootstrap, self.n_windows), dtype=np.float64)
        # For each window, we will grab n_blocks of blocked data of length autocorr_nsteps
        n_boot_samples = n_blocks * autocorr_nsteps
        # We must redefine ones_n vector to reflect boot_n_tot
        n_boot_tot = n_boot_samples.sum()
        ones_n = np.matrix(np.ones(n_boot_tot), dtype=np.float32).T


        def task_gen():
            
            if __debug__:
                checkset = set()
            for lb in xrange(0, self.n_bootstrap, batch_size):
                ub = min(self.n_bootstrap, lb+batch_size)
          
                if __debug__:
                    checkset.update(set(xrange(lb,ub)))

                args = ()
                kwargs = dict(lb=lb, ub=ub, ones_m=ones_m, ones_n=ones_n, bias_mat=self.bias_mat,
                              n_samples=self.n_samples, n_boot_samples=n_boot_samples, n_boot_sample_diag=n_boot_sample_diag, n_boot_tot=n_boot_tot, 
                              n_windows=self.n_windows, autocorr_nsteps=autocorr_nsteps, xweights=-logweights_actual[1:],
                              binbounds=binbounds, all_data_N=self.all_data_N)
                log.info("Sending job batch (from bootstrap sample {} to {})".format(lb, ub))
                yield (_bootstrap, args, kwargs)


        log.info("Beginning {} bootstrap iterations".format(self.n_bootstrap))
        # Splice together results into final array of densities
        for future in self.work_manager.submit_as_completed(task_gen(), queue_size=self.max_queue_len):
            logweights_slice, neglogpdist_N_slice, lb, ub = future.get_result(discard=True)
            log.info("Receiving result")
            logweights_boot[lb:ub, :] = logweights_slice
            log.info("this boot weights: {}".format(logweights_slice))
            if self.fmt=='phi':
                neglogpdist_N_boot[lb:ub, :] = neglogpdist_N_slice
            del logweights_slice

        # Get SE from bootstrapped samples
        logweights_boot_mean = logweights_boot.mean(axis=0)
        logweights_se = np.sqrt(logweights_boot.var(axis=0))
        print('logweights (boot mean): {}'.format(logweights_boot_mean))
        print('logweights: {}'.format(logweights_actual))
        print('se: {}'.format(logweights_se))
        np.savetxt('err_logweights.dat', logweights_se, fmt='%3.6f')

        if self.fmt == 'phi':
            # Get bootstrap errors for -log(Pv(N))
            neglogpdist_N_boot_mean = neglogpdist_N_boot.mean(axis=0)
            neglogpdist_N_se = np.sqrt(neglogpdist_N_boot.var(axis=0))
            #embed()
            pdist_N = gen_pdist(self.all_data_N, self.bias_mat, self.n_samples, logweights_actual, binbounds)
            pdist = gen_pdist(self.all_data, self.bias_mat, self.n_samples, logweights_actual, binbounds)
            pdist_N /= (pdist_N * np.diff(binbounds)).sum()
            pdist /= (pdist_N * np.diff(binbounds)).sum()
            neglogpdist_N = -np.log(pdist_N)
            neglogpdist = -np.log(pdist)

            ## Print some stdout
            print('-ln(P(N)) (boot mean): {}'.format(neglogpdist_N_boot_mean))
            print('-ln(P(N)) (all data): {}'.format(neglogpdist_N))
            print('-ln(P(N)) se: {}'.format(neglogpdist_N_se))

            arr = np.dstack((binbounds[:-1]+np.diff(binbounds)/2.0, neglogpdist_N))
            arr = arr.squeeze()
            
            np.savetxt('neglogpdist_N.dat', arr, fmt='%3.6f')
            arr = np.dstack((binbounds[:-1]+np.diff(binbounds)/2.0, neglogpdist))
            arr = arr.squeeze()
            np.savetxt('neglogpdist.dat', arr, fmt='%3.6f')
            np.savetxt('err_neglogpdist_N.dat', neglogpdist_N_se, fmt='%3.6f')

if __name__=='__main__':
    WHAMmer().main()
