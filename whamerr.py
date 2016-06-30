from __future__ import division, print_function

import numpy as np
from matplotlib import pyplot
import argparse
import logging
from mdtools import dr
import scipy.integrate
from scipy.optimize import fmin_bfgs
import pymbar
import time

from mdtools import ParallelTool

from wham import gen_U_nm, kappa, grad_kappa, gen_pdist

import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rcParams.update({'axes.labelsize': 30})
mpl.rcParams.update({'xtick.labelsize': 18})
mpl.rcParams.update({'ytick.labelsize': 18})
mpl.rcParams.update({'axes.titlesize': 36})
#mpl.rcParams.update({'titlesize': 42})

log = logging.getLogger('mdtools.whamerr')


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


# phidat is datareader (could have different number of values for each ds)
# phivals is (nphi, ) array of phi values (in kT)
def _bootstrap(lb, ub, ones_m, ones_n, bias_mat, n_samples, n_sample_diag,
               n_tot, n_windows, autocorr_nsteps):

    np.random.seed()
    block_size = ub - lb
    logweights_ret = np.zeros((block_size, n_windows), dtype=np.float64)

    # Will only contain uncorrelated, bootstrap samples from biasMat
    bias_mat_boot = np.zeros((n_tot, n_windows), dtype=np.float64)
    xweights = np.zeros(n_windows-1)

    for batch_num in xrange(block_size):
        # Subsample bias_mat
        i_start = 0
        i_start_sub = 0
        for i, n_sample in enumerate(n_samples):
            # the entire bias matrix for this window, shape (nsample, nwindows)
            bias_mat_window = bias_mat[i_start:i_start+n_sample, :]
            rand_start = np.random.randint(autocorr_nsteps)
            # grab every uncorrelated data point - from a random start position
            bias_mat_window_uncorr = bias_mat_window[rand_start::autocorr_nsteps]

            n_pts_uncorr = n_sample // autocorr_nsteps
            # Number of data points after decorrelating matrix
            # should be n_sample/autocorr_nsteps, if not throw out first data point
            # Hackish for when n_sample is not divisible by autocorr length
            if (bias_mat_window_uncorr.shape[0] > n_pts_uncorr): 
                bias_mat_window_uncorr = bias_mat_window_uncorr[1:, :]
                assert bias_mat_window_uncorr.shape[0] == n_pts_uncorr

            # Indices of data points to grap from uncorrelated bias matrix
            #    These are sampled *with* replacement
            #    This comprises our uncorrelated, bootstrap sample for the current window
            indices = np.random.randint(n_pts_uncorr, size=n_pts_uncorr)

            # Append our uncorrelated, bootstrap sample for this window to our bias matrix
            bias_mat_boot[i_start_sub:i_start_sub+n_pts_uncorr, :] = bias_mat_window_uncorr[indices, :]
            i_start += n_sample
            i_start_sub += n_pts_uncorr

        myargs = (bias_mat_boot, n_sample_diag, ones_m, ones_n, n_tot)
        logweights_ret[batch_num, 1:] = -fmin_bfgs(kappa, xweights, fprime=grad_kappa, args=myargs, disp=False)        

    return (logweights_ret, lb, ub)


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

        self.beta = 1

        self.n_bootstrap = None

        self.autocorr = None
        self.ts = None

        self.dr = dr
        self.output_filename = None
    
    # Total number of samples - sum of n_samples from each window
    @property
    def n_tot(self):
        return self.n_samples.sum()
    
    
    def add_args(self, parser):
        
        sgroup = parser.add_argument_group('(Binless) WHAM/MBAR error options')
        sgroup.add_argument('input', metavar='INPUT', type=str, nargs='+',
                            help='Input file names - Must be in ps units!')
        sgroup.add_argument('--fmt', type=str, choices=['phi', 'xvg'], default='phi',
                            help='Format of input data files:  \'phi\' for phiout.dat; \
                            \'xvg\' for XVG type files (i.e. from alchemical GROMACS sims)')
        sgroup.add_argument('-b', '--start', type=int, default=0,
                            help='first timepoint (in ps) - default is first available time point')  
        sgroup.add_argument('-e', '--end', type=int, default=None,
                            help='last timepoint (in ps) - default is last available time point')
        sgroup.add_argument('-T', metavar='TEMP', type=float,
                            help='betaert Phi values to kT, for TEMP (K)')
        sgroup.add_argument('--bootstrap', type=int, default=1000,
                            help='Number of bootstrap samples to perform')   
        sgroup.add_argument('--autocorr', '-ac', type=float, default=1.0,
                            help='Autocorrelation time (in ps); this can be \
                            a single float, or one for each window') 
        sgroup.add_argument('--logweights', type=float, 
                            help='(optional) previously calculated logweights file for INDUS simulations - \
                            if \'phi\' format option also supplied, this will calculate the Pv(N) (and Ntwid). \
                            For \'xvg\' formats, this will calculate the probability distribution of whatever \
                            variable has been umbrella sampled')


    def process_args(self, args):

        self.fmt = args.fmt
        self.n_windows = 0
        try:
            for i, infile in enumerate(args.input):
                log.info("loading {}th input: {}".format(i, infile))

                if self.fmt == 'phi': 
                    self.dr.loadPhi(infile) 
                elif self.fmt == 'xvg': 
                    self.dr.loadXVG(infile)
                self.n_windows += 1
        except:
            raise IOError("Error: Unable to successfully load inputs")


        self.beta = 1
        if args.T:
            self.beta /= (args.T * 8.3144598e-3)

        # Number of bootstrap samples to perform
        self.n_bootstrap = args.bootstrap

        self.autocorr = self._parse_autocorr(args.autocorr)

        self.unpack_data(args.start, args.end)

    # TODO: Parse lists as well
    def _parse_autocorr(self, autocorr):
        return autocorr

    # Note: sets self.ts, as well
    def unpack_data(self, start, end):
        if self.fmt == 'phi':
            self._unpack_phi_data(start, end)

        elif self.fmt == 'xvg':
            self._unpack_xvg_data(start, end)

    # Put all data points into N dim vector
    def _unpack_phi_data(self, start, end=None):

        self.all_data = np.array([], dtype=np.float64)
        self.all_data_N = np.array([]).astype(int)
        self.n_samples = np.array([]).astype(int)

        for i, (ds_name, ds) in enumerate(self.dr.datasets.iteritems()):
            if self.ts == None:
                self.ts = ds.ts
            else:
                np.testing.assert_almost_equal(self.ts, ds.ts)
            data = ds.data[start:end]
            dataframe = np.array(data['$\~N$'], dtype=np.float64)
            dataframe_N = np.array(data['N']).astype(int)

            self.n_samples = np.append(self.n_samples, dataframe.shape[0])

            self.all_data = np.append(self.all_data, dataframe)
            self.all_data_N = np.append(self.all_data_N, dataframe_N)

        self.bias_mat = np.zeros((self.n_tot, self.n_windows), dtype=np.float64)

        # Ugh !
        for i, (ds_name, ds) in enumerate(self.dr.datasets.iteritems()):
            self.bias_mat[:, i] = self.beta*(0.5*ds.kappa*(self.all_data-ds.Nstar)**2 + ds.phi*self.all_data) 

    # Put all data points into N dim vector
    def _unpack_xvg_data(self, start, end=None):

        self.all_data = np.array([], dtype=np.float64)
        self.n_samples = np.array([], dtype=np.int32)
        self.bias_mat = None

        for i, (ds_name, ds) in enumerate(self.dr.datasets.iteritems()):
            if self.ts == None:
                self.ts = ds.ts
            else:
                np.testing.assert_almost_equal(self.ts, ds.ts)
            dataframe = np.array(ds.data[start:end][0])
            bias = self.beta*np.array(ds.data[start:end][1:], dtype=np.float64) # biased values for all windows
            self.n_samples = np.append(self.n_samples, dataframe.shape[0])
            self.all_data = np.append(self.all_data, dataframe)

            if self.bias_mat is None:
                self.bias_mat = bias
            else:
                self.bias_mat = np.vstack((self.bias_mat, bias))

        
    def go(self):

        n_workers = self.work_manager.n_workers or 1
        batch_size = self.n_bootstrap // n_workers
        if self.n_bootstrap % n_workers != 0:
            batch_size += 1
        log.info("batch size: {}".format(batch_size))

        logweights_boot = np.zeros((self.n_bootstrap, self.n_windows), dtype=np.float64)

        # 2*autocorr length (ps), divided by step size (also in ps)
        autocorr_nsteps = int(2*self.autocorr/self.ts)
        log.info("autocorr length: {} ps".format(self.autocorr))
        log.info("data time step: {} ps".format(self.ts))
        log.info("autocorr nsteps: {}".format(autocorr_nsteps))

        # Diagonal is fractional n_samples for each window - should be 
        #    float because future's division function
        uncorr_n_samples = self.n_samples // autocorr_nsteps
        uncorr_ntot = uncorr_n_samples.sum()
        n_sample_diag = np.matrix( np.diag(uncorr_n_samples / uncorr_ntot), dtype=np.float64 )

        ones_m = np.matrix(np.ones(self.n_windows,), dtype=np.float64).transpose()
        ones_n = np.matrix(np.ones(uncorr_ntot), dtype=np.float64).transpose()

        def task_gen():
            
            if __debug__:
                checkset = set()
            for lb in xrange(0, self.n_bootstrap, batch_size):
                ub = min(self.n_bootstrap, lb+batch_size)
          
                if __debug__:
                    checkset.update(set(xrange(lb,ub)))

                args = ()
                kwargs = dict(lb=lb, ub=ub, ones_m=ones_m, ones_n=ones_n, bias_mat=self.bias_mat,
                              n_samples=self.n_samples, n_sample_diag=n_sample_diag, n_tot=uncorr_ntot, 
                              n_windows=self.n_windows, autocorr_nsteps=autocorr_nsteps)
                log.info("Sending job batch (from bootstrap sample {} to {})".format(lb, ub))
                yield (_bootstrap, args, kwargs)


        # Splice together results into final array of densities
        for future in self.work_manager.submit_as_completed(task_gen(), queue_size=self.max_queue_len):
            logweights_slice, lb, ub = future.get_result(discard=True)
            logweights_boot[lb:ub, :] = logweights_slice
            del logweights_slice

        # Redefine these for entire dataset
        ones_n = np.matrix(np.ones(self.n_tot), dtype=np.float64).transpose()
        n_sample_diag = np.matrix( np.diag(self.n_samples / self.n_tot), dtype=np.float64 )

        myargs = (self.bias_mat, n_sample_diag, ones_m, ones_n, self.n_tot)
        xweights = np.zeros(self.n_windows-1)
        logweights_actual = -fmin_bfgs(kappa, xweights, fprime=grad_kappa, args=myargs)
        logweights_actual = np.append(0, logweights_actual)

        # Get SE from bootstrapped samples
        logweights_boot_mean = logweights_boot.mean(axis=0)
        logweights_se = np.sqrt(logweights_boot.var(axis=0))

        ## Get the probability distribution (s)
        if self.fmt == 'phi':
            data_range = (0, self.all_data_N.max()+1)
            binbounds_N, pdist_N = gen_pdist(self.all_data_N, self.bias_mat, self.n_samples, logweights_boot_mean, data_range, data_range[1])
            pdist_N /= pdist_N.sum()
            neglogpdist_N = -np.log(pdist_N)
            arr = np.dstack((binbounds_N[:-1]+np.diff(binbounds_N)/2.0, neglogpdist_N))
            arr = arr.squeeze()
            np.savetxt('neglogpdist_N.dat', arr)

        data_range = (0, self.all_data.max()+1)
        binbounds, pdist = gen_pdist(self.all_data, self.bias_mat, self.n_samples, logweights_boot_mean, data_range, data_range[1])
        pdist /= pdist.sum()
        neglogpdist = -np.log(pdist)

        print('logweights (boot mean): {}'.format(logweights_boot_mean))
        print('logweights: {}'.format(logweights_actual))
        print('se: {}'.format(logweights_se))

        arr = np.dstack((binbounds[:-1]+np.diff(binbounds)/2.0, neglogpdist))
        arr = arr.squeeze()
        np.savetxt('logweights.dat', logweights_actual, fmt='%3.6f')
        np.savetxt('err_logweights.dat', logweights_se, fmt='%3.6f')
        np.savetxt('neglogpdist.dat', arr)
        #print('logweights from bootstrap: {}'.format(logweights_boot))

if __name__=='__main__':
    WHAMmer().main()
