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


# phidat is datareader (could have different number of values for each ds)
# phivals is (nphi, ) array of phi values (in kT)
def _bootstrap(lb, ub, phivals, phidat, autocorr_nsteps, start=0, end=None):

    np.random.seed()
    block_size = ub - lb
    ntwid_ret = np.zeros((len(phidat), block_size), dtype=np.float32)
    integ_ntwid_ret = np.zeros((len(phidat), block_size), dtype=np.float32)
    n_ret = np.zeros_like(ntwid_ret)
    integ_n_ret = np.zeros_like(integ_ntwid_ret)

    # For each bootstrap sample...
    for batch_num in xrange(block_size):
        # For each data set (i.e. INDUS window)
        for i, (ds_name, ds) in enumerate(phidat.items()):
            rand_start = np.random.randint(autocorr_nsteps[i])
            ntwid_sample = np.array(ds.data[start:end]['$\~N$'])
            n_sample = np.array(ds.data[start:end]['N'])

            # Only grab data points after correlation length
            ntwid_sample = ntwid_sample[rand_start::autocorr_nsteps[i]]
            n_sample = n_sample[rand_start::autocorr_nsteps[i]]
            assert ntwid_sample.shape[0] == n_sample.shape[0]

            # Indices of bootstrap sampling - random (with replacement)
            boot_indices = np.random.randint(ntwid_sample.shape[0], size=ntwid_sample.shape[0])
            bootstrap_ntwid = ntwid_sample[boot_indices]
            bootstrap_n = n_sample[boot_indices]
            ntwid_ret[i,batch_num] = bootstrap_ntwid.mean()

            n_reweight = np.exp(-phivals[i]*(bootstrap_n - bootstrap_ntwid))

            n_ret[i,batch_num] = (bootstrap_n*n_reweight).mean() / (n_reweight).mean()

        integ_ntwid_ret[1:,batch_num] = scipy.integrate.cumtrapz(ntwid_ret[:,batch_num], phivals)
        integ_n_ret[1:,batch_num] = scipy.integrate.cumtrapz(n_ret[:,batch_num], phivals)

    return (ntwid_ret, integ_ntwid_ret, n_ret, integ_n_ret, lb, ub)


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
        self.autocorr = None
        self.ts = None

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
        sgroup.add_argument('--mode', choices=['ntwid', 'n'], default='ntwid',
                            help='Integrate and calculate <Ntwid> for \phi Ntwid ensemble (ntwid, default) or <N> \
                                  for \phi N ensemble using reweighting. (n)\
                                  Default ntwid.') 

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

        autocorr = []
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

                # Estimate autocorrelation time from entire data set
                autocorr.append(self.ts * np.ceil(pymbar.timeseries.integratedAutocorrelationTime(ds.data[start:]['$\~N$'])))
        except:
            raise IOError("Error: Unable to successfully load inputs")

        self.autocorr = np.array(autocorr)
        #print(self.autocorr)

        self.conv = 1
        if args.T:
            self.conv /= (args.T * 8.3144598e-3)

        # Number of bootstrap samples to perform
        self.bootstrap = args.bootstrap

        self.output_filename = args.outfile

        self.plotN = args.plotN
        self.plotInteg = args.plotInteg

        self.mode = args.mode

    def go(self):

        phidat = self.dr.datasets
        # value of \beta \phi for each window (if temp provided) or just \phi (in kj/mol if no temp provided)
        phivals = np.zeros(len(phidat), dtype=np.float64)

        ntwid_boot = np.zeros((len(phidat), self.bootstrap), dtype=np.float32)
        n_boot = np.zeros_like(ntwid_boot)
        integ_ntwid_boot = np.zeros((len(phidat), self.bootstrap), dtype=np.float32)
        integ_n_boot = np.zeros_like(integ_ntwid_boot)


        for i, (ds_name, ds) in enumerate(phidat.items()):
            phivals[i] = ds.phi * self.conv


        autocorr_nsteps = (2*self.autocorr/self.ts).astype(int)
        log.info('autocorr nsteps: {}'.format(autocorr_nsteps))

        n_workers = self.work_manager.n_workers or 1
        batch_size = self.bootstrap // n_workers
        if self.bootstrap % n_workers != 0:
            batch_size += 1

        log.info("batch size: {}".format(batch_size))

        def task_gen():
            
            if __debug__:
                checkset = set()
            for lb in xrange(0, self.bootstrap, batch_size):
                ub = min(self.bootstrap, lb+batch_size)
          
                if __debug__:
                    checkset.update(set(xrange(lb,ub)))

                args = ()
                kwargs = dict(lb=lb, ub=ub, phivals=phivals, phidat=phidat, autocorr_nsteps=autocorr_nsteps, 
                              start=self.start, end=self.end)
                log.info("Sending job batch (from bootstrap sample {} to {})".format(lb, ub))
                yield (_bootstrap, args, kwargs)


        # Splice together results into final array of densities
        for future in self.work_manager.submit_as_completed(task_gen(), queue_size=self.max_queue_len):
            ntwid_slice, integ_ntwid_slice, n_slice, integ_n_slice, lb, ub = future.get_result(discard=True)
            ntwid_boot[:, lb:ub] = ntwid_slice
            n_boot[:, lb:ub] = n_slice
            integ_ntwid_boot[:, lb:ub] = integ_ntwid_slice
            integ_n_boot[:, lb:ub] = integ_n_slice
            del ntwid_slice, integ_ntwid_slice, n_slice, integ_n_slice

        ntwid_se = np.sqrt(ntwid_boot.var(axis=1))
        n_se = np.sqrt(n_boot.var(axis=1))
        integ_ntwid_se = np.sqrt(integ_ntwid_boot.var(axis=1))
        integ_n_se = np.sqrt(integ_n_boot.var(axis=1))

        log.info("Var shape: {}".format(ntwid_se.shape))

        # phi  <ntwid>'  \int <ntwid>'  <n>  \int <n>
        out_actual = np.zeros((len(phidat), 5), dtype=np.float64)
        out_actual[:, 0] = phivals

        for i, (ds_name, ds) in enumerate(phidat.items()):
            np.testing.assert_almost_equal(ds.phi*self.conv, out_actual[i, 0])
            ntwid_all = np.array(ds.data[self.start:self.end]['$\~N$'])
            n_all = np.array(ds.data[self.start:self.end]['N'])
            out_actual[i, 1] = ntwid_all.mean()
            n_reweight = np.exp(-phivals[i]*(n_all-ntwid_all))
            out_actual[i, 3] = (n_all*n_reweight).mean() / (n_reweight).mean()

        out_actual[1:, 2] = scipy.integrate.cumtrapz(out_actual[:, 1], phivals)
        out_actual[1:, 4] = scipy.integrate.cumtrapz(out_actual[:, 3], phivals)

        log.info("outputting data to ntwid_err.dat, integ_err.dat, autocorr_len.dat, and {}".format(self.output_filename))
        np.savetxt('ntwid_err.dat', ntwid_se, fmt='%1.4e')
        np.savetxt('n_err.dat', n_se, fmt='%1.4e')
        np.savetxt('integ_ntwid_err.dat', integ_ntwid_se, fmt='%1.4e')
        np.savetxt('integ_n_err.dat', integ_n_se, fmt='%1.4e')
        np.savetxt('autocorr_len.dat', self.autocorr, fmt='%1.4f')
        np.savetxt(self.output_filename, out_actual, fmt='%1.4f')

        if self.plot:
            if self.mode == 'ntwid':
                n_dat = out_actual[:,1]
                n_err = ntwid_se
                n_integ_dat = out_actual[:,2]
                n_integ_err = integ_ntwid_se
                ylab_n = r"$\langle \~N \rangle'\_{\phi}$"
                ylab_n_integ = r"$\int_0^{\phi} \langle \~N \rangle'_{\phi'} d \phi'$"
            elif self.mode == 'n':
                n_dat = out_actual[:,3]
                n_err = n_se
                n_integ_dat = out_actual[:,4]
                n_integ_err = integ_n_se
                ylab_n = r"$\langle N \rangle_\phi$"
                ylab_n_integ = r"$\int_0^\phi \langle N \rangle_{\phi'} d \phi'$"

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
