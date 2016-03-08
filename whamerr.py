from __future__ import division, print_function

import numpy as np
from matplotlib import pyplot
import argparse
import logging
from mdtools import dr
import scipy.integrate
from scipy.optimize import fmin_bfgs
import time

from mdtools import ParallelTool

from wham import gen_U_nm, kappa, grad_kappa

import matplotlib as mpl

mpl.rcParams.update({'axes.labelsize': 30})
mpl.rcParams.update({'xtick.labelsize': 18})
mpl.rcParams.update({'ytick.labelsize': 18})
mpl.rcParams.update({'axes.titlesize': 36})
#mpl.rcParams.update({'titlesize': 42})

log = logging.getLogger('mdtools.whamerr')


## TODO: Eventually betaert this to just do all MBAR analysis (for phiout.dat and XVG)

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
               n_tot, n_windows):

    np.random.seed()
    block_size = ub - lb
    logweights_ret = np.zeros((block_size, n_windows), dtype=np.float64)
    bias_mat_boot = np.zeros_like(bias_mat)
    xweights = np.zeros(n_windows-1)

    for batch_num in xrange(block_size):
        # Subsample bias_mat
        i_start = 0
        for n_sample in n_samples:
            indices = np.random.randint(n_sample, size=n_sample) + i_start
            i_end = i_start+n_sample
            bias_mat_boot[i_start:i_end, :] = bias_mat[indices, :]
            i_start += n_sample

        myargs = (bias_mat_boot, n_sample_diag, ones_m, ones_n, n_tot)
        logweights_ret[batch_num, 1:] = -fmin_bfgs(kappa, xweights, fprime=grad_kappa, args=myargs, disp=False)        

    return (logweights_ret, lb, ub)


class WHAMmer(ParallelTool):
    prog='WHAM/MBAR analysis'
    description = '''\
Perform WHAM/MBAR analysis on datasets. Also perform bootstrapping

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

        # bias for each window
        # shape: (n_tot, n_windows)
        #    Each row contains ith observation at *each* biasing window
        #    Each column corresponds to a single umbrella window
        self.bias_mat = None

        # (n_windows,) array of number of samples in each window
        self.n_samples = None

        self.beta = 1

        self.n_bootstrap = None

        self.dr = dr
        self.output_filename = None
        
    @property
    def n_tot(self):
        return self.n_samples.sum()
    
    
    def add_args(self, parser):
        
        sgroup = parser.add_argument_group('(Binless) WHAM/MBAR error options')
        sgroup.add_argument('input', metavar='INPUT', type=str, nargs='+',
                            help='Input file names - Assumed to be blocked/autocorrelation removed!')
        sgroup.add_argument('--fmt', type=str, choices=['phi', 'xvg'], default='phi',
                            help='Format of input data files:  \'phi\' for phiout.dat; \
                            \'xvg\' for XVG type files (i.e. from alchemical GROMACS sims)')
        sgroup.add_argument('-b', '--start', type=int, default=0,
                            help='first timepoint (in ps) - default is first available time point')  
        sgroup.add_argument('-e', '--end', type=int, default=None,
                            help='last timepoint (in ps) - default is last available time point')
        sgroup.add_argument('-T', metavar='TEMP', type=float,
                            help='betaert Phi values to kT, for TEMP (K)')
        sgroup.add_argument('--bootstrap', type=int, default=10000,
                            help='Number of bootstrap samples to perform')      


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
            self.beta /= (args.T * 8.314462e-3)

        # Number of bootstrap samples to perform
        self.n_bootstrap = args.bootstrap

        self.unpack_data(args.start, args.end)

    def unpack_data(self, start, end):
        if self.fmt == 'phi':
            self._unpack_phi_data(start, end)

        elif self.fmt == 'xvg':
            self._unpack_xvg_data(start, end)

    # Put all data points into N dim vector
    def _unpack_phi_data(self, start, end=None):

        self.all_data = np.array([], dtype=np.float64)
        self.n_samples = np.array([], dtype=np.int32)

        for i, (ds_name, ds) in enumerate(self.dr.datasets.iteritems()):
            dataframe = np.array(ds.data[start:end]['$\~N$'])
            self.n_samples = np.append(self.n_samples, dataframe.shape[0])
            self.all_data = np.append(self.all_data, dataframe)

        self.bias_mat = np.zeros((self.n_tot, self.n_windows), dtype=np.float64)

        # Ugh !
        for i, (ds_name, ds) in enumerate(self.dr.datasets.iteritems()):
            self.bias_mat[:, i] = np.exp( -self.beta*(0.5*ds.kappa*(self.all_data-ds.Nstar)**2 + ds.phi*self.all_data) )

    # Put all data points into N dim vector
    def _unpack_xvg_data(self, start, end=None):

        self.all_data = np.array([], dtype=np.float64)
        self.n_samples = np.array([], dtype=np.int32)
        self.bias_mat = None

        for i, (ds_name, ds) in enumerate(self.dr.datasets.iteritems()):
            dataframe = np.array(ds.data[start:end][0])
            bias = np.exp(-self.beta*np.array(ds.data[start:end][1:], dtype=np.float64)) # biased values for all windows
            self.n_samples = np.append(self.n_samples, dataframe.shape[0])
            self.all_data = np.append(self.all_data, dataframe)

            if self.bias_mat is None:
                self.bias_mat = bias
            else:
                self.bias_mat = np.vstack((self.bias_mat, bias))

        
    def go(self):

        # Diagonal is fractional n_samples for each window - should be 
        #    float because future's division function
        n_sample_diag = np.matrix( np.diag(self.n_samples / self.n_tot), dtype=np.float64 )

        n_workers = self.work_manager.n_workers or 1
        batch_size = self.n_bootstrap // n_workers
        if self.n_bootstrap % n_workers != 0:
            batch_size += 1
        log.info("batch size: {}".format(batch_size))

        logweights_boot = np.zeros((self.n_bootstrap, self.n_windows), dtype=np.float64)
        ones_m = np.matrix(np.ones(self.n_windows,), dtype=np.float64).transpose()
        ones_n = np.matrix(np.ones(self.all_data.shape[0]), dtype=np.float64).transpose()

        def task_gen():
            
            if __debug__:
                checkset = set()
            for lb in xrange(0, self.n_bootstrap, batch_size):
                ub = min(self.n_bootstrap, lb+batch_size)
          
                if __debug__:
                    checkset.update(set(xrange(lb,ub)))

                args = ()
                kwargs = dict(lb=lb, ub=ub, ones_m=ones_m, ones_n=ones_n, bias_mat=self.bias_mat,
                              n_samples=self.n_samples, n_sample_diag=n_sample_diag, n_tot=self.n_tot, 
                              n_windows=self.n_windows)
                log.info("Sending job batch (from bootstrap sample {} to {})".format(lb, ub))
                yield (_bootstrap, args, kwargs)


        # Splice together results into final array of densities
        for future in self.work_manager.submit_as_completed(task_gen(), queue_size=self.max_queue_len):
            logweights_slice, lb, ub = future.get_result(discard=True)
            logweights_boot[lb:ub, :] = logweights_slice
            del logweights_slice

        myargs = (self.bias_mat, n_sample_diag, ones_m, ones_n, self.n_tot)
        xweights = np.zeros(self.n_windows-1)
        logweights_actual = fmin_bfgs(kappa, xweights, fprime=grad_kappa, args=myargs)
        logweights_actual = np.append(0, -logweights_actual)

        # Get SE from bootstrapped samples
        logweights_se = np.sqrt(logweights_boot.var(axis=0))

        print('logweights: {}'.format(logweights_actual))
        print('se: {}'.format(logweights_se))

        np.savetxt('logweights.dat', logweights_actual, fmt='%3.3f')
        np.savetxt('err_logweights.dat', logweights_se, fmt='%3.3f')
        #print('logweights from bootstrap: {}'.format(logweights_boot))

if __name__=='__main__':
    WHAMmer().main()
