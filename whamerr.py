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

from whamutils import kappa, grad_kappa, hess_kappa, gen_data_logweights, get_neglogpdist

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
def _bootstrap(lb, ub, ones_m, ones_n, uncorr_ones_n, bias_mat, n_samples, uncorr_n_samples,
               uncorr_n_sample_diag, uncorr_n_tot, n_windows, autocorr_blocks, 
               xweights, all_data, all_data_N, boot_fn=None):

    np.random.seed()
    batch_size = ub - lb
    f_k_ret = np.zeros((batch_size, n_windows), dtype=np.float32)
    boot_fn_ret = np.zeros(batch_size, dtype=object)

    # Get the associated bootstrapped (log of the) weights for each datapoint
    logweights_ret = np.zeros((batch_size, uncorr_n_tot))

    # for each bootstrap sample in this batch
    for batch_num in range(batch_size):

        ## Fill up the uncorrelated bias matrix
        boot_uncorr_bias_mat = np.zeros((uncorr_n_tot, n_windows), dtype=np.float32)
        start_idx = 0
        uncorr_start_idx = 0
        boot_indices = np.array([], dtype=int)

        for i, block_size in enumerate(autocorr_blocks):
            this_uncorr_n_sample = uncorr_n_samples[i] # number of rows
            this_n_sample = n_samples[i]
            
            # start indices (rows in bias_mat) for each block for this window
            avail_indices = np.arange(this_n_sample)
            this_indices = start_idx + np.random.choice(avail_indices, size=this_uncorr_n_sample, replace=True)
            boot_indices = np.append(boot_indices, this_indices)
            boot_uncorr_bias_mat[uncorr_start_idx:uncorr_start_idx+this_uncorr_n_sample, :] = bias_mat[this_indices, :]
            
            uncorr_start_idx += this_uncorr_n_sample
            start_idx += this_n_sample

        # WHAM this bootstrap sample

        myargs = (boot_uncorr_bias_mat, uncorr_n_sample_diag, ones_m, uncorr_ones_n, uncorr_n_tot)
        boot_f_k = -np.append(0, minimize(kappa, xweights, method='L-BFGS-B', jac=grad_kappa, args=myargs).x)

        f_k_ret[batch_num,:] = boot_f_k
        
        if boot_fn is not None:
            boot_logweights = gen_data_logweights(boot_uncorr_bias_mat, boot_f_k, uncorr_n_samples)
            #embed()
            boot_fn_ret[batch_num] = boot_fn(all_data, all_data_N, boot_indices, boot_logweights)
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

        self.boot_fn = None
    
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
        sgroup.add_argument('--min-autocorr-time', type=float, default=0,
                            help='The minimum autocorrelation time to use (in ps). Default is no minimum')
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


    def process_args(self, args):

        self.fmt = args.fmt
        self.n_windows = 0
        try:
            for i, infile in enumerate(args.input):
                log.info("loading {}th input: {}".format(i, infile))

                if self.fmt == 'phi': 
                    self.dr.loadPhi(infile) 
                elif self.fmt == 'pmf':
                    self.dr.loadPMF(infile)
                elif self.fmt == 'simple':
                    ds = self.dr.loadSimple(infile)
                self.n_windows += 1
        except:
            raise IOError("Error: Unable to successfully load inputs")
        
        # Ignored if not doing xvg...
        self.for_lmbdas = pd.Index(sorted(self.for_lmbdas))
        self.min_autocorr_time = args.min_autocorr_time

        self.beta = 1
        if args.T:
            self.beta /= (args.T * 8.3144598e-3)

        # Number of bootstrap samples to perform
        self.n_bootstrap = args.bootstrap

        if args.f_k:
            self.start_weights = -np.loadtxt(args.f_k)
            log.info("starting weights: {}".format(self.start_weights))
        
        if args.autocorr_file is not None:
            self._parse_autocorr_list(args.autocorr_file)
        if args.autocorr is not None:
            self._parse_autocorr(args.autocorr)
        self.unpack_data(args.start, args.end, args.skip)

        self.nbins = args.nbins
        
        if args.boot_fn is not None:
            self.boot_fn = get_object(args.boot_fn)
    
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

        elif self.fmt == 'rama':
            self._unpack_rama_data(start, end)

        elif self.fmt == 'simple':
            self._unpack_simple_data(start, end)

        elif self.fmt == 'pmf':
            self._unpack_pmf_data(start, end)


    # Construct bias matrix from data
    def _unpack_simple_data(self, start, end=None):

        self.all_data = np.array([], dtype=np.float32)
        self.n_samples = np.array([]).astype(int)

        if self.autocorr is None:
            do_autocorr = True
            self.autocorr = np.zeros(self.n_windows)
        else:
            do_autocorr = False

        for i, (ds_name, ds) in enumerate(self.dr.datasets.items()):

            if self.ts == None:
                self.ts = []
                
            # Sanity check - every input should have same timestep
            #else:
            #    np.testing.assert_almost_equal(self.ts, ds.ts)
            self.ts.append(ds.ts)
            data = ds.data[start:end]
            dataframe = np.array(data)

            if do_autocorr:
                try:
                    autocorr_len = np.ceil(pymbar.timeseries.integratedAutocorrelationTime(dataframe[:, -1]))
                    self.autocorr[i] = max(ds.ts * autocorr_len, self.min_autocorr_time)
                except:
                    embed()

            self.n_samples = np.append(self.n_samples, dataframe.shape[0])

            if self.bias_mat is None:
                self.bias_mat = self.beta*dataframe.copy()
            else:
                self.bias_mat = np.vstack((self.bias_mat, self.beta*dataframe))

            self.all_data = np.append(self.all_data, dataframe[:,-1])
            
        #if do_autocorr:
        log.info("saving integrated autocorr times (in ps) to 'autocorr.dat'")
        np.savetxt('autocorr.dat', self.autocorr, header='integrated autocorrelation times for each window (in ps)')
        self.ts = np.array(self.ts)
        dr.clearData()

    # Put all data points into N dim vector
    def _unpack_pmf_data(self, start, end=None):

        self.all_data = np.array([], dtype=np.float32) # R dists
        self.n_samples = np.array([]).astype(int)

        if self.autocorr is None:
            do_autocorr = True
            self.autocorr = np.zeros(self.n_windows)
        else:
            do_autocorr = False

        for i, (ds_name, ds) in enumerate(self.dr.datasets.items()):

            if self.ts == None:
                self.ts = []
                
            # Sanity check - every input should have same timestep
            #else:
            #    np.testing.assert_almost_equal(self.ts, ds.ts)
            self.ts.append(ds.ts)
            data = ds.data[start:end]
            dataframe = np.array(data['rstar'])

            if do_autocorr:
                autocorr_len = np.ceil(pymbar.timeseries.integratedAutocorrelationTime(dataframe[:]))
                self.autocorr[i] = max(ds.ts * autocorr_len, self.min_autocorr_time)

            self.n_samples = np.append(self.n_samples, dataframe.shape[0])

            self.all_data = np.append(self.all_data, dataframe)

        self.bias_mat = np.zeros((self.n_tot, self.n_windows), dtype=np.float32)
        
        # Ugh !
        #embed()
        for i, (ds_name, ds) in enumerate(self.dr.datasets.items()):
            self.bias_mat[:, i] = self.beta*(0.5*ds.kappa*(self.all_data-ds.rstar)**2) 
        
        #if do_autocorr:
        log.info("saving integrated autocorr times (in ps) to 'autocorr.dat'")
        np.savetxt('autocorr.dat', self.autocorr, header='integrated autocorrelation times for each window (in ps)')
        self.ts = np.array(self.ts)
        dr.clearData()

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

        for i, (ds_name, ds) in enumerate(self.dr.datasets.items()):

            if self.ts == None:
                self.ts = []
                
            # Sanity check - every input should have same timestep
            #else:
            #    np.testing.assert_almost_equal(self.ts, ds.ts)
            self.ts.append(ds.ts)
            data = ds.data[start:end]
            dataframe = np.array(data['$\~N$'])
            dataframe_N = np.array(data['N']).astype(np.int32)

            if do_autocorr:
                autocorr_len = np.ceil(pymbar.timeseries.integratedAutocorrelationTime(dataframe[:]))
                self.autocorr[i] = max(ds.ts * autocorr_len, self.min_autocorr_time)

            self.n_samples = np.append(self.n_samples, dataframe.shape[0])

            self.all_data = np.append(self.all_data, dataframe)
            self.all_data_N = np.append(self.all_data_N, dataframe_N)


        self.bias_mat = np.zeros((self.n_tot, self.n_windows), dtype=np.float32)
        
        # Ugh !
        for i, (ds_name, ds) in enumerate(self.dr.datasets.items()):
            self.bias_mat[:, i] = self.beta*(0.5*ds.kappa*(self.all_data-ds.Nstar)**2 + ds.phi*self.all_data) 
        
        #if do_autocorr:
        log.info("saving integrated autocorr times (in ps) to 'autocorr.dat'")
        np.savetxt('autocorr.dat', self.autocorr, header='integrated autocorrelation times for each window (in ps)')
        self.ts = np.array(self.ts)
        dr.clearData()
        
    # Put all data points into N dim vector
    def _unpack_xvg_data(self, start, end=None,skip=None):
        raise NotImplementedError

    # Put all data points into N dim vector
    def _unpack_rama_data(self, start, end=None):
        raise NotImplementedError

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

        # Diagonal is fractional uncorr_n_samples for each window - should be 
        #    float because future's division function
        uncorr_n_sample_diag = np.matrix( np.diag(uncorr_n_samples / uncorr_n_tot), dtype=np.float32 )
        n_sample_diag = np.matrix( np.diag(self.n_samples / self.n_tot), dtype=np.float32)
        # Run MBAR once before bootstrapping
        ones_m = uncorr_ones_m = np.matrix(np.ones(self.n_windows,), dtype=np.float32).T
        # (n_tot x 1) ones vector; n_tot = sum(n_k) total number of samples over all windows
        uncorr_ones_n = np.matrix(np.ones(uncorr_n_tot,), dtype=np.float32).T
        ones_n = np.matrix(np.ones(self.n_tot,), dtype=np.float32).T

        if self.start_weights is not None:
            log.info("using initial weights: {}".format(self.start_weights))
            xweights = self.start_weights
        else:
            xweights = np.zeros(self.n_windows)

        assert xweights[0] == 0

        embed()

        #myargs = (uncorr_bias_mat, uncorr_n_sample_diag, uncorr_ones_m, uncorr_ones_n, uncorr_n_tot)
        myargs = (self.bias_mat, n_sample_diag, ones_m, ones_n, self.n_tot)
        log.info("Running MBAR on entire dataset")

        # fmin_bfgs spits out a tuple with some extra info, so we only take the first item (the weights)
        f_k_actual = minimize(kappa, xweights[1:], method='L-BFGS-B', jac=grad_kappa, args=myargs).x
        f_k_actual = np.append(0, f_k_actual)
        log.info("MBAR results on entire dataset: {}".format(f_k_actual))

        np.savetxt('f_k_all.dat', f_k_actual, fmt='%3.6f')
        all_logweights = gen_data_logweights(self.bias_mat, f_k_actual, self.n_samples)
        
        # Now for bootstrapping...
        n_workers = self.work_manager.n_workers or 1
        batch_size = self.n_bootstrap // n_workers
        if self.n_bootstrap % n_workers != 0:
            batch_size += 1
        log.info("batch size: {}".format(batch_size))

        # the bootstrap estimates of free energies wrt window i=0
        f_k_boot = np.zeros((self.n_bootstrap, self.n_windows), dtype=np.float64)
        boot_res = np.zeros(self.n_bootstrap, dtype=object)

        def task_gen():
            
            if __debug__:
                checkset = set()
            for lb in range(0, self.n_bootstrap, batch_size):
                ub = min(self.n_bootstrap, lb+batch_size)
          
                if __debug__:
                    checkset.update(set(range(lb,ub)))

                args = ()
                kwargs = dict(lb=lb, ub=ub, ones_m=ones_m, ones_n=ones_n, uncorr_ones_n=uncorr_ones_n, bias_mat=self.bias_mat,
                              n_samples=self.n_samples, uncorr_n_samples=uncorr_n_samples, uncorr_n_sample_diag=uncorr_n_sample_diag, 
                              uncorr_n_tot=uncorr_n_tot, n_windows=self.n_windows, autocorr_blocks=autocorr_blocks, xweights=-f_k_actual[1:],
                              all_data=self.all_data, all_data_N=self.all_data_N, boot_fn=self.boot_fn)
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
        np.savez_compressed('all_data.dat', logweights=all_logweights, data=self.all_data, data_N=self.all_data_N, bias_mat=self.bias_mat, n_samples=self.n_samples)


        if self.boot_fn is not None:
            np.save('boot_fn_payload.dat', boot_res)

if __name__=='__main__':
    WHAMmer().main()
