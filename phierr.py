from __future__ import division, print_function

import numpy as np
from matplotlib import pyplot
import argparse
import logging
from mdtools import dr
import scipy.integrate
import pymbar
import time

from mdtools import ParallelTool

import matplotlib as mpl
from matplotlib import pyplot as plt


mpl.rcParams.update({'axes.labelsize': 30})
mpl.rcParams.update({'xtick.labelsize': 18})
mpl.rcParams.update({'ytick.labelsize': 18})
mpl.rcParams.update({'axes.titlesize': 36})
#mpl.rcParams.update({'titlesize': 42})

log = logging.getLogger('mdtools.phierr')


# phidat is datareader.datasets (could have different number of values for each ds)
# phivals is (nphi, ) array of phi values (in kT)
def _bootstrap(lb, ub, phivals, phidat, autocorr_nsteps, start=0, end=None):

    np.random.seed()

    # batch_size is the number of independent bootstrap samples for this job
    batch_size = ub - lb

    ntwid_ret = np.zeros((len(phidat), batch_size), dtype=np.float32)
    ntwid_var_ret = np.zeros_like(ntwid_ret)   
    integ_ntwid_ret = np.zeros((len(phidat), batch_size), dtype=np.float32)

    n_ret = np.zeros_like(ntwid_ret)
    n_var_ret = np.zeros_like(ntwid_ret)
    integ_n_ret = np.zeros_like(integ_ntwid_ret)

    # Getting the slopes numerically might have less noise...
    fin_diff_ntwid = np.zeros_like(ntwid_ret)
    fin_diff_n = np.zeros_like(ntwid_ret)

    # For each bootstrap sample...
    for batch_num in range(batch_size):
        # For each data set (i.e. INDUS window)
        for i, ds in enumerate(phidat):
            # Full datasets for ntwid, n
            ntwid_sample = np.array(ds.data[start:end]['$\~N$'])
            n_sample = np.array(ds.data[start:end]['N'])
            assert ntwid_sample.shape[0] == n_sample.shape[0], "Ntwid and N must have same shape for given dataset!"
            
            sample_size = ntwid_sample.shape[0]
            # Cast should be unnecessary, but just to be sure
            # block_size (not to be confused with batch size!) is the size of the bootstrapping 'block' - here, the autocorr length
            block_size = int(autocorr_nsteps[i])
            # Number of blocks of block size autocorr we can take from dataset
            num_blocks = int(sample_size/block_size)

            num_boot_sample = num_blocks * block_size
            ntwid_boot_sample = np.zeros(num_boot_sample)
            n_boot_sample = np.zeros(num_boot_sample)

            # this (minus 1) is the maximum start index from which we can grab a full block from full dataset
            avail_start_indices = sample_size - block_size + 1
            boot_start_indices = np.random.randint(avail_start_indices, size=num_blocks) 

            for k, boot_start_idx in enumerate(boot_start_indices):
                start_idx = k*block_size
                ntwid_boot_sample[start_idx:start_idx+block_size] = ntwid_sample[boot_start_idx:boot_start_idx+block_size]
                n_boot_sample[start_idx:start_idx+block_size] = n_sample[boot_start_idx:boot_start_idx+block_size]

            assert ntwid_boot_sample.shape[0] == n_boot_sample.shape[0]

            ntwid_mean = ntwid_boot_sample.mean()
            ntwid_ret[i,batch_num] = ntwid_mean
            ntwid_var_ret[i,batch_num] = (ntwid_boot_sample**2).mean() - ntwid_mean**2

            n_reweight = np.exp(-phivals[i]*(n_boot_sample - ntwid_boot_sample))
            n_mean = (n_boot_sample*n_reweight).mean() / (n_reweight).mean()
            n_sq_mean = ((n_boot_sample**2)*n_reweight).mean() / (n_reweight).mean()
            n_ret[i,batch_num] = n_mean
            n_var_ret[i,batch_num] = n_sq_mean - n_mean**2

        integ_ntwid_ret[1:,batch_num] = scipy.integrate.cumtrapz(ntwid_ret[:,batch_num], phivals)
        integ_n_ret[1:,batch_num] = scipy.integrate.cumtrapz(n_ret[:,batch_num], phivals)

        fin_diff_ntwid[1:,batch_num] = -np.diff(ntwid_ret[:,batch_num]) / np.diff(phivals)
        fin_diff_n[1:,batch_num] = -np.diff(n_ret[:,batch_num]) / np.diff(phivals)

    return (ntwid_ret, ntwid_var_ret, integ_ntwid_ret, n_ret, n_var_ret, integ_n_ret, fin_diff_ntwid, fin_diff_n, lb, ub)


class Phierr(ParallelTool):
    prog='phi error analysis'
    description = '''\
Perform bootstrap error analysis for phi dataset (N v phi)

This tool calculates the autocorrelation time automagically with PYMBAR for each dataset.
    It's integrated into the rest of the bootstrapping analysis - this program will
    take care of autocorrelated data and provide accurate estimates for standard errors.

This tool supports parallelization (see options below)


-----------------------------------------------------------------------------
Command-line options
-----------------------------------------------------------------------------
'''
    
    def __init__(self):
        super(Phierr,self).__init__()
        
        # Parallel processing by default (this is not actually necessary, but it is
        # informative!)
        self.wm_env.default_work_manager = self.wm_env.default_parallel_work_manager

        self.start = None
        self.end = None

        self.conv = 1

        self.bootstrap = None
        # Autocorrelation time (ps)
        self.autocorr = None
        self.ts = None

        self.phidat = None
        self.phivals = None

        self.dr = dr
        self.output_filename = None

        self.plotN = None
        self.plotInteg = None


    @property
    def plot(self):
        return self.plotN or self.plotInteg
    
        
    
    def add_args(self, parser):
        
        sgroup = parser.add_argument_group('Phi error options')
        sgroup.add_argument('input', metavar='INPUT', type=str, nargs='+',
                            help='Input file names')
        sgroup.add_argument('-b', '--start', type=int, default=0,
                            help='first timepoint (in ps)')  
        sgroup.add_argument('-e', '--end', type=int, default=None,
                            help='last timepoint (in ps) - default is last available time point')
        sgroup.add_argument('-T', metavar='TEMP', type=float,
                            help='convert Phi values to kT, for TEMP (K)')
        sgroup.add_argument('--bootstrap', type=int, default=1000,
                            help='Number of bootstrap samples to perform')      
        sgroup.add_argument('--autocorr', '-ac', type=float, help='Autocorrelation time (in ps); this can be \ '
                            'a single float, or one for each window') 
        sgroup.add_argument('--mode', choices=['ntwid', 'n'], default='ntwid',
                            help='''(Plotting option) integrate and calculate <Ntwid> for \phi Ntwid ensemble (ntwid, default) or <N> \
                                  for phi N ensemble using reweighting. (n)\
                                  Default ntwid. NOTE: This option controls plotting only - this tool automatically outputs
                                  for Ntwid and N''') 

        agroup = parser.add_argument_group('other options')
        agroup.add_argument('--plotN', action='store_true',
                            help='Show Ntwid (or N) v (beta) phi, with calculated error bars')
        agroup.add_argument('--plotInteg', action='store_true',
                            help='Show the integral of Ntwid (or N) v (beta) phi, with calculated error bars')
        agroup.add_argument('--plotSus', action='store_true',
                            help='Show var Ntwid (or N) v (beta) phi')
        agroup.add_argument('-o', '--outfile', default='out.dat',
                            help='Output file to write -ln(Q_\phi/Q_0)')

    def process_args(self, args):

        self.conv = 1
        if args.T:
            self.conv /= (args.T * 8.3144598e-3)

        autocorr = []
        phidat = []
        phivals = []
        self.start = start = args.start
        self.end = end = args.end

        try:
            for i, infile in enumerate(args.input):
                log.info("loading {}th input: {}".format(i, infile))
                ds = self.dr.loadPhi(infile)
                if self.ts == None:
                    self.ts = ds.ts
                else:
                    np.testing.assert_almost_equal(self.ts, ds.ts)
                if args.autocorr:
                    autocorr.append(args.autocorr)
                else:
                    try:
                        dataframe = np.array(ds.data[start:]['$\~N$'])
                    # Estimate autocorrelation time from entire data set
                        autocorr.append(self.ts * np.ceil(pymbar.timeseries.integratedAutocorrelationTime(dataframe, fast=True)))
                    except:
                        autocorr.append(self.ts * 10)

                # save phi val and phi dataset
                phivals.append(ds.phi * self.conv)
                phidat.append(ds)
        except:
            raise IOError("Error: Unable to successfully load inputs")
        #embed()

        phivals = np.array(phivals)
        phidat = np.array(phidat)
        autocorr = np.array(autocorr)
        # Sort based on phi
        # returns *indices*
        sorted_indices = np.argsort(phivals)

        self.autocorr = autocorr[sorted_indices]
        #print(self.autocorr)

        self.phidat = phidat[sorted_indices]
        self.phivals = phivals[sorted_indices]

        # Number of bootstrap samples to perform
        self.bootstrap = args.bootstrap

        self.output_filename = args.outfile

        self.plotN = args.plotN
        self.plotInteg = args.plotInteg

        self.mode = args.mode

    def go(self):

        ntwid_boot = np.zeros((len(self.phidat), self.bootstrap), dtype=np.float64)
        ntwid_var_boot = np.zeros_like(ntwid_boot)
        n_boot = np.zeros_like(ntwid_boot)
        n_var_boot = np.zeros_like(ntwid_boot)
        integ_ntwid_boot = np.zeros((len(self.phidat), self.bootstrap), dtype=np.float64)
        integ_n_boot = np.zeros_like(integ_ntwid_boot)

        # numerical slopes
        ntwid_fin_diff_boot = np.zeros_like(ntwid_boot)
        n_fin_diff_boot = np.zeros_like(ntwid_boot)

        autocorr_nsteps = (1+2*self.autocorr/self.ts).astype(int)
        log.info('autocorr nsteps: {}'.format(autocorr_nsteps))

        n_workers = self.work_manager.n_workers or 1
        batch_size = self.bootstrap // n_workers
        if self.bootstrap % n_workers != 0:
            batch_size += 1

        log.info("batch size: {}".format(batch_size))

        def task_gen():
            
            if __debug__:
                checkset = set()
            for lb in range(0, self.bootstrap, batch_size):
                ub = min(self.bootstrap, lb+batch_size)
          
                if __debug__:
                    checkset.update(set(range(lb,ub)))

                args = ()
                kwargs = dict(lb=lb, ub=ub, phivals=self.phivals, phidat=self.phidat, autocorr_nsteps=autocorr_nsteps, 
                              start=self.start, end=self.end)
                log.info("Sending job batch (from bootstrap sample {} to {})".format(lb, ub))
                yield (_bootstrap, args, kwargs)


        # Splice together results into final array of densities
        for future in self.work_manager.submit_as_completed(task_gen(), queue_size=self.max_queue_len):
            ntwid_slice, ntwid_var_slice, integ_ntwid_slice, n_slice, n_var_slice, integ_n_slice, ntwid_fin_diff_slice, n_fin_diff_slice, lb, ub = future.get_result(discard=True)
            ntwid_boot[:, lb:ub] = ntwid_slice
            ntwid_var_boot[:, lb:ub] = ntwid_var_slice
            n_boot[:, lb:ub] = n_slice
            n_var_boot[:, lb:ub] = n_var_slice
            integ_ntwid_boot[:, lb:ub] = integ_ntwid_slice
            integ_n_boot[:, lb:ub] = integ_n_slice
            ntwid_fin_diff_boot[:, lb:ub] = ntwid_fin_diff_slice
            n_fin_diff_boot[:, lb:ub] = n_fin_diff_slice
            del ntwid_slice, integ_ntwid_slice, n_slice, n_var_slice, integ_n_slice, ntwid_fin_diff_slice, n_fin_diff_slice

        ntwid_se = ntwid_boot.std(axis=1, ddof=1)
        ntwid_var_se = ntwid_var_boot.std(axis=1, ddof=1)
        n_se = n_boot.std(axis=1, ddof=1)
        n_var_se = n_var_boot.std(axis=1, ddof=1)
        integ_ntwid_se = integ_ntwid_boot.std(axis=1, ddof=1)
        integ_n_se = integ_n_boot.std(axis=1, ddof=1)
        ntwid_fin_diff_se = ntwid_fin_diff_boot.std(axis=1, ddof=1)
        n_fin_diff_se = n_fin_diff_boot.std(axis=1, ddof=1)

        log.info("Var shape: {}".format(ntwid_se.shape))

        ntwid_out_header = "phi   <ntwid>' <d ntwid^2>' d<ntwid>'/dphi  integ(<ntwid>')"
        ntwid_out_actual = np.zeros((len(self.phidat), 9), dtype=np.float64)
        ntwid_out_actual[:, 0] = self.phivals

        n_out_header = "phi   <n>(reweighted) <d n^2>(reweighted) d<n>(reweighted)/dphi  integ(<n>(reweighted))"
        n_out_actual = np.zeros_like(ntwid_out_actual)
        n_out_actual[:, 0] = self.phivals

        for i, ds in enumerate(self.phidat):
            np.testing.assert_almost_equal(ds.phi*self.conv, ntwid_out_actual[i, 0])
            ntwid_all = np.array(ds.data[self.start:self.end]['$\~N$'])
            n_all = np.array(ds.data[self.start:self.end]['N'])

            # Average Ntwid under Ntwid*phi ensemble (sampled)
            ntwid_mean = ntwid_all.mean()
            ntwid_out_actual[i, 1] = ntwid_mean
            ntwid_out_actual[i, 2] = (ntwid_all**2).mean() - ntwid_mean**2
            
            n_reweight = np.exp(-self.phivals[i]*(n_all-ntwid_all))
            # Average N under N*phi ensemble (reweighted)
            n_mean = (n_all*n_reweight).mean() / n_reweight.mean()
            n_sq_mean = ((n_all**2)*n_reweight).mean() / n_reweight.mean()
            n_out_actual[i, 1] = n_mean
            n_out_actual[i, 2] = n_sq_mean - n_mean**2


        ntwid_out_actual[1:, 3] = -np.diff(ntwid_out_actual[:, 1]) / np.diff(self.phivals)
        n_out_actual[1:, 3] = -np.diff(n_out_actual[:, 1]) / np.diff(self.phivals)
        ntwid_out_actual[1:, 4] = scipy.integrate.cumtrapz(ntwid_out_actual[:, 1], self.phivals)
        n_out_actual[1:, 4] = scipy.integrate.cumtrapz(n_out_actual[:, 1], self.phivals)

        log.info("outputting data to {}, as well as bootstrapped standard errors".format(self.output_filename))
        np.savetxt('ntwid_err.dat', ntwid_se, fmt='%1.4e')
        np.savetxt('ntwid_var_err.dat', ntwid_var_se, fmt='%1.4e')
        np.savetxt('n_err.dat', n_se, fmt='%1.4e')
        np.savetxt('n_var_err.dat', n_var_se, fmt='%1.4e')
        np.savetxt('integ_ntwid_err.dat', integ_ntwid_se, fmt='%1.4e')
        np.savetxt('integ_n_err.dat', integ_n_se, fmt='%1.4e')
        np.savetxt('ntwid_fin_diff_err.dat', ntwid_fin_diff_se, fmt='%1.4e')
        np.savetxt('n_fin_diff_err.dat', n_fin_diff_se, fmt='%1.4e')
        np.savetxt('autocorr_len.dat', self.autocorr, fmt='%1.4f')
        np.savetxt('ntwid_{}'.format(self.output_filename), ntwid_out_actual, fmt='%1.4f', header=ntwid_out_header)
        np.savetxt('n_{}'.format(self.output_filename), n_out_actual, fmt='%1.4f', header=n_out_header)

        if self.plot:
            if self.conv == 1:
                beta_pref = ''
            else:
                beta_pref = r"\beta"
            if self.mode == 'ntwid':
                n_dat = out_actual[:,1]
                n_err = ntwid_se
                n_integ_dat = out_actual[:,2]
                n_integ_err = integ_ntwid_se
                ylab_n = r"$" + r"\langle \~N \rangle_{\phi}$"
                ylab_n_integ = r"$" + beta_pref + r"\int_0^{\phi} \langle \~N \rangle_{\phi'} d \phi'$"

            elif self.mode == 'n':
                n_dat = out_actual[:,3]
                n_err = n_se
                n_integ_dat = out_actual[:,4]
                n_integ_err = integ_n_se
                ylab_n = r"$" + r"\langle N \rangle_\phi$"
                ylab_n_integ = r"$" + beta_pref + r"\int_0^\phi \langle N \rangle_{\phi'} d \phi'$"

            beta_phi = out_actual[:,0]
            if self.conv == 1:
                xlab = r'$\phi$ (kJ/mol)'
            else:
                xlab = r'$\beta\phi$ ($k_B T$)'
            
            if self.plotN:
                plt.errorbar(beta_phi, n_dat, yerr=n_err, fmt='-o', linewidth=6, elinewidth=6, capsize=6, markersize=12)
                plt.xlabel(xlab)
                plt.ylabel(ylab_n)
                plt.show()
            if self.plotInteg:
                plt.errorbar(beta_phi, n_integ_dat, yerr=n_integ_err, fmt='-o', linewidth=6, elinewidth=6, capsize=6, markersize=12)
                plt.xlabel(xlab)
                plt.ylabel(ylab_n_integ)
                plt.show()


if __name__=='__main__':
    Phierr().main()
