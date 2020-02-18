import numpy as np
from mdtools import dr
import pymbar

from IPython import embed

import logging

DTYPE = np.float64

log = logging.getLogger('mdtools.WHAMDataExtractor')


class WHAMDataExtractor:

    def __init__(self, infiles, fmt='phi', auxinfiles=None, autocorr=None, start=0, end=None, beta=1):
        self.dr = dr
        self.fmt = fmt
        self.n_windows = 0

        self.beta = beta
        # All data points, from all windows
        # shape: (N_tot,)
        self.all_data = None
        # Optionally hold extra auxiliary data
        self.all_data_aux = None

        # Number of (possibly correlated) samples in each dataset
        # shape: (N_windows, )
        self.n_samples = None
        
        try:
            for i, infile in enumerate(infiles):
                log.info("loading {}th input: {}".format(i, infile))

                if self.fmt == 'phi': 
                    self.dr.loadPhi(infile) 
                elif self.fmt == 'simple':
                    this_auxfile = None
                    if auxinfiles is not None:
                        this_auxfile = auxinfiles[i]
                    self.dr.loadSimple(infile, aux_filename=this_auxfile)
                self.n_windows += 1
        except:
            raise IOError("Error: Unable to successfully load inputs")

        if autocorr is not None:
            # Assume a file - try to read in tau's
            if type(autocorr) is str:
                try:
                    self.autocorr = np.loadtxt(autocorr)
                except:
                    raise IOError("Error: Unable to load autocorr file {}".format(autocorr))
            elif type(autocorr) is float:
                self.autocorr = np.zeros(self.n_windows)
                self.autocorr[:] = autocorr

            self.calc_autocorr = False

        else:
            self.autocorr = None
            self.calc_autocorr = True

        # Timestep in ps for each dataset - assumed to be the same across all
        self.ts = None

        self.unpack_data(start, end)


    # 2*autocorr length (ps), divided by step size (also in ps) - the block_size(s)
    #    i.e. the number of subsequent data points for each window over which data is uncorrelated
    @property
    def autocorr_blocks(self):
        if self.autocorr is not None:
            return np.ceil(1+2*self.autocorr/self.ts)

    # 1/autocorr_blocks - multiply n_samples in each window to get the number of uncorrelated samples
    @property
    def stat_ineff(self):
        if self.autocorr_blocks is not None:
            return 1 / self.autocorr_blocks

    # Total number of (possibly correlated) samples
    @property
    def n_tot(self):
        return self.n_samples.sum()

    @property
    def uncorr_n_samples(self):
        return np.floor(self.n_samples * self.stat_ineff).astype(int)

    @property
    def uncorr_n_tot(self):
        return self.uncorr_n_samples.sum()

    @property
    def n_sample_diag(self):
        return np.diag(self.n_samples/self.uncorr_n_tot)

    @property
    def uncorr_n_sample_diag(self):
        return np.diag(self.uncorr_n_samples/self.uncorr_n_tot)

    # Shape: (n_windows,)
    #   contains stat ineff for each window
    @property
    def ones_m(self):
        return self.stat_ineff

    # shape (n_tot,)
    #   contains stat ineff for all points in each window
    @property
    def ones_n(self):
        ones_n = np.zeros(self.n_tot)
        cum_n_samples = np.append(0, np.cumsum(self.n_samples))

        for i in range(self.n_windows):
            start_idx = cum_n_samples[i]
            end_idx = cum_n_samples[i+1]
            ones_n[start_idx:end_idx] = self.stat_ineff[i]

        return ones_n

    #
    @property
    def uncorr_ones_m(self):
        return np.ones(self.n_windows)

    @property
    def uncorr_ones_n(self):
        return np.ones(self.uncorr_n_tot)

    # Note: sets self.ts, as well
    def unpack_data(self, start, end):
        if self.fmt == 'phi':
            self._unpack_phi_data(start, end)

        elif self.fmt == 'simple':
            self._unpack_simple_data(start, end)


    # Put all data points into N dim vector
    def _unpack_phi_data(self, start, end=None):

        self.all_data = np.array([], dtype=DTYPE)
        self.all_data_aux = np.array([], dtype=DTYPE)
        self.n_samples = np.array([]).astype(int)

        # If autocorr not set, set it ouselves
        if self.calc_autocorr:
            self.autocorr = np.zeros(self.n_windows)

        for i, (ds_name, ds) in enumerate(self.dr.datasets.items()):

            if self.ts == None:
                self.ts = ds.ts
            # Sanity check - every input should have same timestep
            else:
                np.testing.assert_almost_equal(self.ts, ds.ts)
            
            data = ds.data[start:end]
            dataframe = np.array(data['$\~N$'])
            dataframe_N = np.array(data['N']).astype(np.int32)

            if self.calc_autocorr:
                autocorr_len = np.ceil(pymbar.timeseries.integratedAutocorrelationTime(dataframe[:]))
                self.autocorr[i] = ds.ts * autocorr_len

            self.n_samples = np.append(self.n_samples, dataframe.shape[0])

            self.all_data = np.append(self.all_data, dataframe)
            self.all_data_aux = np.append(self.all_data_aux, dataframe_N)


        self.bias_mat = np.zeros((self.n_tot, self.n_windows), dtype=np.float32)

        # Ugh !
        for i, (ds_name, ds) in enumerate(self.dr.datasets.items()):
            self.bias_mat[:, i] = self.beta*(0.5*ds.kappa*(self.all_data-ds.Nstar)**2 + ds.phi*self.all_data) 
        
        log.info("saving integrated autocorr times (in ps) to 'autocorr.dat'")
        np.savetxt('autocorr.dat', self.autocorr, header='integrated autocorrelation times for each window (in ps)')
        
        dr.clearData()

    # Construct bias matrix from data
    def _unpack_simple_data(self, start, end=None):

        self.all_data = np.array([], dtype=np.float32)
        self.all_data_aux = np.array([], dtype=DTYPE)
        self.n_samples = np.array([]).astype(int)
        self.bias_mat = None

        # If autocorr not set, set it ouselves
        if self.calc_autocorr:
            self.autocorr = np.zeros(self.n_windows)
        
        for i, (ds_name, ds) in enumerate(self.dr.datasets.items()):
            
            if self.ts == None:
                self.ts = []
                
            # Sanity check - every input should have same timestep
            #else:
            #    np.testing.assert_almost_equal(self.ts, ds.ts)
            self.ts.append(ds.ts)
            data = ds.data[start:end]
            dataframe = np.array(data)

            if self.calc_autocorr:
                autocorr_len = np.ceil(pymbar.timeseries.integratedAutocorrelationTime(dataframe[:, 0]))
                self.autocorr[i] = ds.ts * autocorr_len

            self.n_samples = np.append(self.n_samples, dataframe.shape[0])

            if self.bias_mat is None:
                self.bias_mat = self.beta*dataframe.copy()
            else:
                self.bias_mat = np.vstack((self.bias_mat, self.beta*dataframe))

            self.all_data = np.append(self.all_data, dataframe[:,i])
            self.all_data_aux = np.append(self.all_data_aux, ds.aux_data[start:end])
            
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

    # Generator to produce observed data points for each window, 
    #   along with their biases, for comparing consensus histogram
    #   generated from uwham
    def gen_obs_data(self):

        cum_n_samples = np.append(0, np.cumsum(self.n_samples))

        for i in range(self.n_windows):
            start_idx = cum_n_samples[i]
            end_idx = cum_n_samples[i+1]

            yield (self.all_data[start_idx:end_idx], self.bias_mat[start_idx:end_idx, i])

    
