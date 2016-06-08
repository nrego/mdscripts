from __future__ import division, print_function

import numpy as np
from matplotlib import pyplot
import argparse
import logging
from mdtools import dr
import scipy.integrate
import time

from mdtools import ParallelTool

import matplotlib as mpl

mpl.rcParams.update({'axes.labelsize': 30})
mpl.rcParams.update({'xtick.labelsize': 18})
mpl.rcParams.update({'ytick.labelsize': 18})
mpl.rcParams.update({'axes.titlesize': 36})
#mpl.rcParams.update({'titlesize': 42})

log = logging.getLogger('mdtools.phierr')


# phidat is datareader (could have different number of values for each ds)
# phivals is (nphi, ) array of phi values (in kT)
def _bootstrap(lb, ub, phivals, phidat):

    np.random.seed()
    block_size = ub - lb
    ntwid_ret = np.zeros((len(phidat), block_size), dtype=np.float32)
    integ_ret = np.zeros((len(phidat), block_size), dtype=np.float32)

    for batch_num in xrange(block_size):
        for i, (ds_name, ds) in enumerate(phidat.items()):

            ntwidarr = np.array(ds.data[:]['$\~N$'])
            bootstrap = np.random.choice(ntwidarr, size=len(ntwidarr))
            ntwid_ret[i,batch_num] = bootstrap.mean()

        integ_ret[1:,batch_num] = scipy.integrate.cumtrapz(ntwid_ret[:,batch_num], phivals)

    return (ntwid_ret, integ_ret, lb, ub)


class Phierr(ParallelTool):
    prog='phi error analysis'
    description = '''\
Perform bootstrap error analysis for phi dataset (N v phi)

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

        self.dr = dr
        self.output_filename = None
        
    
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
        sgroup.add_argument('--bootstrap', type=int, default=10000,
                            help='Number of bootstrap samples to perform')      

        agroup = parser.add_argument_group('other options')
        agroup.add_argument('-o', '--outfile', default='out.dat',
                        help='Output file to write -ln(Q_\phi/Q_0)')

    def process_args(self, args):
        try:
            for i, infile in enumerate(args.input):
                log.info("loading {}th input: {}".format(i, infile))
                self.dr.loadPhi(infile)
        except:
            raise IOError("Error: Unable to successfully load inputs")

        self.start = args.start
        self.end = args.end

        self.conv = 1
        if args.T:
            self.conv /= (args.T * 8.3144598e-3)

        # Number of bootstrap samples to perform
        self.bootstrap = args.bootstrap

        self.output_filename = args.outfile

    def go(self):

        phidat = self.dr.datasets
        phivals = np.zeros(len(phidat), dtype=np.float64)

        ntwid_boot = np.zeros((len(phidat), self.bootstrap), dtype=np.float32)
        integ_boot = np.zeros((len(phidat), self.bootstrap), dtype=np.float32)

        for i, (ds_name, ds) in enumerate(phidat.items()):
            phivals[i] = ds.phi * self.conv

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
                kwargs = dict(lb=lb, ub=ub, phivals=phivals, phidat=phidat)
                log.info("Sending job batch (from bootstrap sample {} to {})".format(lb, ub))
                yield (_bootstrap, args, kwargs)


        # Splice together results into final array of densities
        for future in self.work_manager.submit_as_completed(task_gen(), queue_size=self.max_queue_len):
            ntwid_slice, integ_slice, lb, ub = future.get_result(discard=True)
            ntwid_boot[:, lb:ub] = ntwid_slice
            integ_boot[:, lb:ub] = integ_slice
            del ntwid_slice, integ_slice

        ntwid_se = np.sqrt(ntwid_boot.var(axis=1))
        integ_se = np.sqrt(integ_boot.var(axis=1))

        log.info("Var shape: {}".format(ntwid_se.shape))

        out_actual = np.zeros((len(phidat), 3), dtype=np.float64)
        out_actual[:, 0] = phivals

        for i, (ds_name, ds) in enumerate(phidat.items()):
            np.testing.assert_almost_equal(ds.phi*self.conv, out_actual[i, 0])
            ntwidarr = np.array(ds.data[:]['$\~N$'])
            out_actual[i, 1] = ntwidarr.mean()

        out_actual[1:, 2] = scipy.integrate.cumtrapz(out_actual[:, 1], phivals)

        log.info("outputting data to ntwid_err.dat, integ_err.dat, and {}".format(self.output_filename))
        np.savetxt('ntwid_err.dat', ntwid_se, fmt="%1.8e")
        np.savetxt('integ_err.dat', integ_se, fmt="%1.8e")
        np.savetxt(self.output_filename, out_actual, fmt="%1.3f")


if __name__=='__main__':
    Phierr().main()
