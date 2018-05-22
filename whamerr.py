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

#from IPython import embed


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
def _bootstrap(lb, ub, ones_m, ones_n, ones_n_uncorr, bias_mat, n_samples, n_uncorr_samples,
               n_uncorr_sample_diag, n_uncorr_tot, n_windows, autocorr_blocks, 
               xweights, binbounds, all_data_N):

    np.random.seed()
    batch_size = ub - lb
    logweights_ret = np.zeros((batch_size, n_windows), dtype=np.float32)

    # for INDUS datasets
    if binbounds is not None and all_data_N is not None:
        neglogpdist_N_ret = np.zeros((batch_size, binbounds.shape[0]-1))
    else:
        neglogpdist_N_ret = None

    # for each bootstrap sample in this batch
    for batch_num in xrange(batch_size):
        ## Fill up the uncorrelated bias matrix
        boot_uncorr_bias_mat = np.zeros((n_uncorr_tot, n_windows), dtype=np.float32)
        start_idx = 0
        uncorr_start_idx = 0
        #embed()
        # Now gather the effective reduced bias_mat by accounting for autocorrelation -
        #   for each window i, average those rows into continuous blocks of size autocorr_nstep
        for i, block_size in enumerate(autocorr_blocks):
            this_n_uncorr_sample = n_uncorr_samples[i] # number of rows
            this_n_sample = n_samples[i]
            
            # start indices (rows in bias_mat) for each block for this window
            avail_indices = np.arange(this_n_sample)
            this_indices = start_idx + np.random.choice(avail_indices, size=this_n_uncorr_sample, replace=True)

            boot_uncorr_bias_mat[uncorr_start_idx:uncorr_start_idx+this_n_uncorr_sample, :] = bias_mat[this_indices, :]
            
            uncorr_start_idx += this_n_uncorr_sample
            start_idx += this_n_sample

        # WHAM this bootstrap sample

        myargs = (boot_uncorr_bias_mat, n_uncorr_sample_diag, ones_m, ones_n_uncorr, n_uncorr_tot)
        boot_weights = -np.array(fmin_bfgs(kappa, xweights, fprime=grad_kappa, args=myargs)[0])
        logweights_ret[batch_num, 1:] = boot_weights
        del boot_uncorr_bias_mat

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

        # shape (n_tot,)
        # Du/Dl for each window at each timepoint (i.e. Delta U=U1 - U0 when simple linear interp)
        # Then the *bias* this observation feels at window i is self.dudl * lambda_i
        self.dudl = None

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
        self.unpack_data(args.start, args.end, args.skip)

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
    def unpack_data(self, start, end, skip):
        if self.fmt == 'phi':
            self._unpack_phi_data(start, end)

        elif self.fmt == 'xvg':
            self._unpack_xvg_data(start, end, skip)

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
    def _unpack_xvg_data(self, start, end=None,skip=None):

        self.all_data = None
        self.n_samples = np.array([], dtype=np.int32)
        self.bias_mat = None

        if self.autocorr is None:
            do_autocorr = True
            self.autocorr = np.zeros(self.n_windows)
        else:
            do_autocorr = False

        do_skip = False
        if skip is not None:
            do_skip = True

        for i, (ds_name, ds) in enumerate(self.dr.datasets.iteritems()):
            #embed()

            log.info("Unpacking {}th dataset ({:s})".format(i, ds_name))

            # Only set the ts once, for first ds (and check that it's the same for subsequent)
            # This is also the time to check for the step size if we are skipping data, by
            #   dividing skip by the timestep
            if self.ts == None:
                self.ts = ds.ts
                if do_skip:
                    step = int(skip / self.ts)
                    log.info("only grabbing data every {} ps ({} steps)".format(skip, step))
                else:
                    step = 1 # by default take every data point
                    log.info("grabbing data every step ({} ps)".format(self.ts))
            else:
                np.testing.assert_almost_equal(self.ts, ds.ts)

            ## quick test that dataset makes sense - DU's for its own lambda should be all zeros
            arr_self = ds.data[ds.lmbda]
            np.testing.assert_array_almost_equal(arr_self, np.zeros_like(arr_self), decimal=5)


            ## Sanity checks done. Now grab this window's data (i.e. its du's) and poss. calc the iact
            dataframe = np.array(ds.data[start:end:step][self.for_lmbdas], dtype=np.float32)

            if do_autocorr:
                log.info("    Calculating autocorrelation time...")
                
                n_samples = dataframe.shape[0]
                max_autocorr_len = n_samples // 50 # Can only accurately get tau if (n_samples > 50*tau)
                iact = 0 # initial guess
                # Get autocorr time for this lambda window by looking at its dU's for its adjacent windows
                for k in [i-1, i+1]:
                    
                    if k < 0:
                        continue
                    try:
                        # iact in units of *steps*, *NOT* time
                        curr_iact = np.ceil(pymbar.timeseries.integratedAutocorrelationTime(dataframe[:,k], fast=True))
                        if curr_iact > iact:
                            iact = curr_iact
                    except:
                        continue
                # self.autocorr is IACT in **ps**
                self.autocorr[i] = max(self.ts * iact, self.min_autocorr_time)
                log.info("      Tau={} ps ({}) steps".format(self.autocorr[i], iact))
                

            bias = self.beta*dataframe # biased values for all windows
            self.n_samples = np.append(self.n_samples, dataframe.shape[0])
            
            this_dudl = self.beta * np.array(ds.dhdl[start:end:step], dtype=np.float32)
            if self.dudl is None:
                self.dudl = this_dudl
            else:
                self.dudl = np.vstack((self.dudl, this_dudl))
            if self.all_data is None:
                self.all_data = dataframe
            else:
                self.all_data = np.vstack((self.all_data, dataframe))

            if self.bias_mat is None:
                self.bias_mat = bias
            else:
                self.bias_mat = np.vstack((self.bias_mat, bias))

        ## Finished unpacking all data windows - now rearrange the bias matrix, clean up, etc
        self.dudl = np.squeeze(self.dudl)

        # Fix bias matrix so each column lam_i is now U_i-U_0; i.e. bias w.r.t window 0
        col0 = self.bias_mat[:, 0].copy()
        #for i in xrange(self.n_windows):
        #   self.bias_mat[:, i] = self.bias_mat[:, i] - col0
        
        if do_autocorr:
            log.info("saving integrated autocorr times (in ps) to 'autocorr.dat'")
            np.savetxt('autocorr.dat', self.autocorr, header='integrated autocorrelation times for each window (in ps)')
        dr.clearData()
        
    def go(self):
        
        # 2*autocorr length (ps), divided by step size (also in ps) - the block_size(s)
        #    i.e. the number of subsequent data points for each window over which data is uncorrelated
        autocorr_blocks = np.ceil(1+2*self.autocorr/self.ts).astype(int)

        log.info("Tau for each window: {} ps".format(self.autocorr))
        log.info("data time step: {} ps".format(self.ts))
        log.info("autocorr nsteps: {}".format(autocorr_blocks))

        # Number of complete blocks for each sample window - the effective 
        #    number of uncorrelated samples for each window
        uncorr_n_samples = self.n_samples // autocorr_blocks
        remainders = self.n_samples % autocorr_blocks
        # Total number of effective uncorrelated samples
        uncorr_n_tot  = uncorr_n_samples.sum()
        assert uncorr_n_tot <= self.n_tot

        # Diagonal is fractional n_uncorr_samples for each window - should be 
        #    float because future's division function
        uncorr_n_sample_diag = np.matrix( np.diag(uncorr_n_samples / uncorr_n_tot), dtype=np.float32 )
        n_sample_diag = np.matrix( np.diag(self.n_samples / self.n_tot), dtype=np.float32)
        # Run MBAR once before bootstrapping
        ones_m = uncorr_ones_m = np.matrix(np.ones(self.n_windows,), dtype=np.float32).T
        # (n_tot x 1) ones vector; n_tot = sum(n_k) total number of samples over all windows
        uncorr_ones_n = np.matrix(np.ones(uncorr_n_tot,), dtype=np.float32).T
        ones_n = np.matrix(np.ones(self.n_tot,), dtype=np.float32).T


        ## Fill up the uncorrelated bias matrix
        uncorr_bias_mat = np.zeros((uncorr_n_tot, self.n_windows), dtype=np.float32)
        start_idx = 0
        uncorr_start_idx = 0
        
        uncorr_data = np.zeros((uncorr_n_tot, ), dtype=np.float32)
        # Now gather the effective reduced bias_mat by accounting for autocorrelation -
        #   for each window i, average those rows into continuous blocks of size autocorr_nstep
        for i, block_size in enumerate(autocorr_blocks):
            this_n_uncorr_sample = uncorr_n_samples[i] # number of rows
            this_n_sample = self.n_samples[i]

            # Start offset so the number of uncorrelated data points lines up
            remainder = remainders[i]
            #embed()
            uncorr_data[uncorr_start_idx:uncorr_start_idx+this_n_uncorr_sample] = self.all_data[start_idx+remainder:start_idx+this_n_sample:block_size]

            uncorr_data_slice = self.bias_mat[start_idx+remainder:start_idx+this_n_sample:block_size, :]
            uncorr_bias_mat[uncorr_start_idx:uncorr_start_idx+this_n_uncorr_sample] = uncorr_data_slice

            uncorr_start_idx += this_n_uncorr_sample
            start_idx += this_n_sample

        if self.start_weights is not None:
            log.info("using initial weights: {}".format(self.start_weights))
            xweights = self.start_weights
        else:
            xweights = np.zeros(self.n_windows)

        assert xweights[0] == 0

        myargs = (uncorr_bias_mat, uncorr_n_sample_diag, uncorr_ones_m, uncorr_ones_n, uncorr_n_tot)
        #myargs = (self.bias_mat, n_sample_diag, ones_m, ones_n, self.n_tot)
        log.info("Running MBAR on entire dataset")
        
        # fmin_bfgs spits out a tuple with some extra info, so we only take the first item (the weights)
        logweights_actual = fmin_bfgs(kappa, xweights[1:], fprime=grad_kappa, args=myargs)[0]
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

        # the bootstrap estimates of free energies wrt window i=0
        logweights_boot = np.zeros((self.n_bootstrap, self.n_windows), dtype=np.float64)
        assert logweights_actual[0] == 0
        def task_gen():
            
            if __debug__:
                checkset = set()
            for lb in xrange(0, self.n_bootstrap, batch_size):
                ub = min(self.n_bootstrap, lb+batch_size)
          
                if __debug__:
                    checkset.update(set(xrange(lb,ub)))

                args = ()
                kwargs = dict(lb=lb, ub=ub, ones_m=ones_m, ones_n=ones_n, ones_n_uncorr=uncorr_ones_n, bias_mat=self.bias_mat,
                              n_samples=self.n_samples, n_uncorr_samples=uncorr_n_samples, n_uncorr_sample_diag=uncorr_n_sample_diag, n_uncorr_tot=uncorr_n_tot, 
                              n_windows=self.n_windows, autocorr_blocks=autocorr_blocks, xweights=-logweights_actual[1:],
                              binbounds=binbounds, all_data_N=self.all_data_N)
                log.info("Sending job batch (from bootstrap sample {} to {})".format(lb, ub))
                yield (_bootstrap, args, kwargs)


        log.info("Beginning {} bootstrap iterations".format(self.n_bootstrap))
        # Splice together results into final array of densities
        for future in self.work_manager.submit_as_completed(task_gen(), queue_size=self.max_queue_len):
            logweights_slice, neglogpdist_N_slice, lb, ub = future.get_result(discard=True)
            log.info("Receiving result")
            logweights_boot[lb:ub, :] = logweights_slice
            log.debug("this boot weights: {}".format(logweights_slice))
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
   
            pdist_N = gen_pdist(self.all_data_N, self.bias_mat, self.n_samples, logweights_actual, binbounds)
            pdist = gen_pdist(self.all_data, self.bias_mat, self.n_samples, logweights_actual, binbounds)
            pdist_N /= (pdist_N * np.diff(binbounds)).sum()
            pdist /= (pdist * np.diff(binbounds)).sum()
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
