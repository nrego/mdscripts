"""
Static data reader class for analysing phi time series data from simulations

nrego
"""
from __future__ import print_function, division; __metaclass__ = type

import numpy as np
import pandas
from matplotlib import pyplot
import logging
import re
import linecache
import collections

log = logging.getLogger(__name__)


def normhistnd(hist, binbounds):
    '''Normalize the N-dimensional histogram ``hist`` with corresponding
    bin boundaries ``binbounds``.  Modifies ``hist`` in place and returns
    the normalization factor used.'''

    diffs = np.append(np.diff(binbounds), 0)

    assert diffs.shape == hist.shape
    normfac = (hist * diffs[0]).sum()

    hist /= normfac
    return normfac

def extractFloat(string):
    return map(float, re.findall(r"[-+]?\d*\.\d+|\d+", string))


class DataSet:

    def __init__(self):
        self.title = None
        self.data = None

    @property
    def shape(self):
        return self.data.shape

    def plot(self, ylim=None, start=0, block=1):
        raise NotImplemetedError

    def getMean(self, start=0, axis=1):
        return self.data[start:, axis].mean()

    def __repr__(self):
        return "DataSet <{!s:}>".format(self.title)


class PhiDataSet(DataSet):

    def __init__(self, filename, corr_len=1):
        super(PhiDataSet, self).__init__()

        # Assume output file from umbrella.conf
        self.kappa = extractFloat(linecache.getline(filename, 1)).pop()
        self.Nstar = extractFloat(linecache.getline(filename, 2)).pop()
        self.phi = extractFloat(linecache.getline(filename, 3)).pop() # Ugh
        linecache.clearcache()
        data = np.loadtxt(filename)
        log.debug('Datareader {} reading input file {}'.format(self, filename))
        self.data = pandas.DataFrame(data[::corr_len, 1:], index=data[::corr_len, 0],
                                     columns=['N', r'$\~N$'])
        self.title = filename

    # Block is block size of data
    def printOut(self, filename, start, end=None, block=1):

        n_obs = len(self.data[start:end])
        data = np.zeros((n_obs, 3), dtype=np.float32)
        data[:,0] = self.data[start:end].index
        data[:,1:] = np.array(self.data[start:end])

        data = data[1:] # generally cut off first datapoint

        n_block = int(n_obs/block)
        obs_prop = np.zeros((n_block, 3), dtype=np.float32)

        for i in xrange(n_block):
            ibeg = i*block
            iend = ibeg + block
            obs_prop[i] = data[ibeg:iend].mean(axis=0)

        header = '''kappa =    {:1.3f} kJ/mol
        NStar =    {:1.3f}
        mu =    {:1.3f} kJ/mol
        t (ps)        N       NTwiddle'''.format(self.kappa, self.Nstar, self.phi)

        np.savetxt(filename, obs_prop, header=header, fmt="%1.6f        %2.6f        %2.6f")

    def blockAvg(self, start, end=None, outfile=None):

        data = np.array(self.data[start:end]['$\~N$'])
        data = data[1:]
        #data = ds
        data_var = data.var()
        n_obs = len(data)  # Total number of observations

        #blocks = (np.power(2, xrange(int(np.log2(n_obs))))).astype(int)
        # Block size
        blocks = np.arange(1,len(data)/2+1,1)

        n_blocks = len(blocks)

        block_vals = np.zeros((n_blocks, 3))
        block_vals[:, 0] = blocks.copy()

        block_ctr = 0

        for block in blocks:
            n_block = int(n_obs/block)
            obs_prop = np.zeros(n_block)

            for i in xrange(n_block):
                ibeg = i*block
                iend = ibeg + block
                obs_prop[i] = data[ibeg:iend].mean()

            block_vals[block_ctr, 1] = obs_prop.mean()
            block_vals[block_ctr, 2] = obs_prop.var() / (n_block-1)

            block_ctr += 1

        return block_vals

    def plot(self, start=0, ylim=None, block=1, end=None):
        pandas.rolling_mean(self.data[start:end:10], window=block).plot()

        mean = self.getMean(start=start, end=end)
        line = pyplot.hlines(mean, start, self.shape[0])
        line.set_label('mean: {:.2f}'.format(mean))

        if (ylim is not None):
            pyplot.ylim(ylim)

    def getRange(self, start=0, end=None):
        rng = self.data[start:end].max() - self.data[start:end:10].min()

        return rng['$\~N$']

    def max(self, start=0, end=None):
        return self.data[start:end].max()

    def min(self, start=0, end=None):
        return self.data[start:end].min()

    def getMean(self, start=0, bphi=1, end=None):
        #return self.data[start:].mean()[1]
        #N = self.data[start:]['N']
        Ntwid = self.data[start:end]['$\~N$']
        #numer = (N*np.exp(bphi*(Ntwid-N)))
        #denom = (np.exp(bphi*(Ntwid-N)))

        #return numer.mean() / denom.mean()
        return Ntwid.mean()

    def getSecondMom(self, start=0, bphi=1, end=None):
        NtwidSq = (self.data[start:end]['$\~N$'])**2

        return NtwidSq.mean()

    def getVar(self, start=0, bphi=1, end=None):
        N_avg = self.getMean(start, bphi, end=end)

        #N = self.data[start:]['N']
        Ntwid = self.data[start:end]['$\~N$']
        #numer = (N-N_avg)**2 * np.exp(bphi*(Ntwid-N))
        #denom = np.exp(bphi*(Ntwid-N))

        #return numer.mean() / denom.mean()
        return ((Ntwid - N_avg)**2).mean()

    def getHist(self, start=0, nbins=50, end=None):
        return np.histogram(self.data[start:end]['$\~N$'], bins=nbins)


# For free energy calcs
class XvgDataSet(DataSet):

    def __init__(self, filename, corr_len=1):
        super(XvgDataSet, self).__init__()

        self.lbdas = []
        self.temp = None

        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                elif line.startswith('@'):
                    if line.find('T =') != -1:
                        self.temp = extractFloat(line)[0]
                    if line.find('s0') != -1:
                        self.lbdas.append(extractFloat(line[-1]))
                    elif line.find('s1') != -1 or line.find('s2') != -1:
                        self.lbdas.append(extractFloat(line[-1]))
                else:
                    break

        #self.title = title
        data = np.loadtxt(filename, comments=['#', '@'])
        log.debug('Datareader {} reading input file {}'.format(self, filename))
        self.data = pandas.DataFrame(data[::corr_len, 1:], index=data[::corr_len, 0],
                                     columns=[])

    def printOut(self, filename, start, end=None, block=1):

        n_obs = len(self.data[start:end])
        data = np.zeros((n_obs, 3), dtype=np.float32)
        data[:,0] = self.data[start:end].index
        data[:,1:] = np.array(self.data[start:end])

        data = data[1:] # generally cut off first datapoint

        n_block = int(n_obs/block)
        obs_prop = np.zeros((n_block, 3), dtype=np.float32)

        for i in xrange(n_block):
            ibeg = i*block
            iend = ibeg + block
            obs_prop[i] = data[ibeg:iend].mean(axis=0)

        header = '''kappa =    {:1.3f} kJ/mol
        NStar =    {:1.3f}
        mu =    {:1.3f} kJ/mol
        t (ps)        N       NTwiddle'''.format(self.kappa, self.Nstar, self.phi)

        np.savetxt(filename, obs_prop, header=header, fmt="%1.6f        %2.6f        %2.6f")

    def plot(self, ylim=None, start=0, block=1):
        pyplot.plot(self.data[:, 0], self.data[:, 1], label=self.name)
        #pyplot.plot(self.data[:, 0], self.data[:, 2], label=r'$beta P$')

        pyplot.xlabel(r'$kJ/mol$')

        if (ylim is not None):
            pyplot.ylim(ylim)


class DataReader:
    '''Global class for handling datasets, etc'''
    datasets = collections.OrderedDict()
    ts = None
    start = 0.0
    mean = -1
    phi = -1

    '''Load phiout.dat files'''
    @classmethod
    def loadPhi(cls, filename, corr_len=1):
        ds = PhiDataSet(filename, corr_len)
        return cls._addSet(ds)

    @classmethod
    def loadXVG(cls, filename, corr_len=1):
        ds = XvgDataSet(filename, corr_len)
        return cls._addSet(ds)

    @classmethod
    def _addSet(cls, ds):
        cls.datasets[ds.title] = ds
        return ds

    @classmethod
    def plot(cls, ylim=None, start=0, end=None, block=1):
        for title, dataset in cls.datasets.items():
            dataset.plot(ylim=ylim, start=start, end=end, block=block)

    @classmethod
    def plotHistAll(cls, start=0, end=None, nbins=50):
        total_array = np.array([])
        for title, dataset in cls.datasets.iteritems():
            total_array = np.append(total_array, dataset.data[start:end]['$\~N$'])
            data = dataset.data[start:end]['$\~N$']
            #pyplot.hist(np.array(data), bins=nbins, normed=True, label="phi: {} kj/mol".format(dataset.phi))

        pyplot.legend()
        counts, centers = np.histogram(total_array, bins=nbins)
        centers = np.diff(centers)/2.0 + centers[:-1]
        pyplot.bar(centers, counts, width=np.diff(centers)[0])

        retarr = np.zeros((centers.size, 2),dtype=np.float32)
        retarr[:,0] = centers
        retarr[:,1] = counts

        return retarr


    @classmethod
    def clearData(cls):
        del cls.datasets
        cls.datasets = collections.OrderedDict()

    @staticmethod
    def show():
        pyplot.legend()
        pyplot.show()


_titleString = [r'$N(dU/d\lambda | \lambda = {:.2f})$',
                r'$P(\Delta U(\lambda = {:.2f}) | \lambda = {:.2f}))$']

dr = DataReader
