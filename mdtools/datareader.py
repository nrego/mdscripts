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

from IPython import embed


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

    # Return dataset's timestep (in ps)
    @property
    def ts(self):
        if self.data is None:
            return -1
        else:
            return np.diff(self.data.index)[0]

    
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
        #self.kappa = extractFloat(linecache.getline(filename, 1)).pop()
        #self.Nstar = extractFloat(linecache.getline(filename, 2)).pop()
        #self.phi = extractFloat(linecache.getline(filename, 3)).pop() # Ugh
        # Scan first few lines of phiout.dat file, looking for 'phi', 'Nstar', 
        #   and 'kappa'
        self._scan_header_dat(filename)
        #linecache.clearcache()
        data = np.loadtxt(filename)
        log.debug('Datareader {} reading input file {}'.format(self, filename))
        self.data = pandas.DataFrame(data[::corr_len, 1:], index=data[::corr_len, 0],
                                     columns=['N', r'$\~N$'])
        self.title = filename
        log.debug('....DONE;  kappa={}, Nstar={}, phi={}'.format(self.kappa,self.Nstar,self.phi))

    def _scan_header_dat(self, filename):
        kappa_found = False
        Nstar_found = False
        phi_found = False
        with open(filename, 'r') as fin:
            for line in fin:
                if 'kappa' in line:
                    self.kappa = extractFloat(line).pop()
                    kappa_found = True
                elif 'NStar' in line:
                    self.Nstar = extractFloat(line).pop()
                    Nstar_found = True
                elif 'mu' in line:
                    self.phi = extractFloat(line).pop()
                    phi_found = True
                if (kappa_found and Nstar_found and phi_found):
                    break

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
        rm = self.data[start:end].rolling(window=block).sum()
        rm.plot()
        mean = self.getMean(start=start, end=end)
        line = pyplot.hlines(mean, start, rm.index[-1], zorder=3)
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
        #return ((Ntwid - N_avg)**2).mean()
        return Ntwid.var(ddof=1)

    def getHist(self, start=0, nbins=50, end=None):
        return np.histogram(self.data[start:end]['$\~N$'], bins=nbins)



# For free energy calcs
class XvgDataSet(DataSet):

    def __init__(self, filename, corr_len=1):
        super(XvgDataSet, self).__init__()

        # Value of lambda for each window
        self.lmbdas = []
        self.lmbda = None
        self.temp = None

        self.dhdl = None

        self.title=filename
        self.header = ''
        with open(filename, 'r') as f:
            
            for line in f:
                #print(line)
                if line.startswith('#'):
                    continue
                elif line.startswith('@'):
                    self.header += line
                    if line.find('T =') != -1:
                        self.temp = extractFloat(line)[0]
                        self.lmbda = extractFloat(line)[1]
                    if line.find('s0') != -1:
                        continue
                    elif re.findall('s[0-9]+', line):
                        self.lmbdas.append(extractFloat(line)[1])
                else:
                    break
        #print(self.lmbdas)
        self.header = self.header.rstrip()
        data = np.loadtxt(filename, comments=['#', '@'])
        log.debug('Datareader {} reading input file {}'.format(self, filename))

        #self.dhdl1 = pandas.DataFrame(data[::corr_len, 1], index=data[::corr_len, 0])
        # Data has biases in kJ/mol !
        self.data = pandas.DataFrame(data[::corr_len, 2:], index=data[::corr_len, 0],
                                     columns=self.lmbdas)

        self.dhdl = pandas.DataFrame(self.data[self.lmbdas[-1]] - self.data[self.lmbdas[0]])

    def blockAvg(self, start=None, end=None, outfile=None):
        blocks = np.arange(1,min(len(self.data)//2, 5000))

        n_blocks = len(blocks)
        
        total_block_vals = np.zeros((len(self.lmbdas), n_blocks, 3))

        for i,key in enumerate(self.lmbdas):

            data = np.array(self.data[start:end][key])
            data = data[1:]
            n_obs = len(data)  # Total number of observations
            data_var = data.var()


            #blocks = (np.power(2, xrange(int(np.log2(n_obs))))).astype(int)
            # Block size

            block_vals = total_block_vals[i]
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

        return total_block_vals

    def printOut(self, filename, start=None, end=None, block=1):

        n_obs = len(self.data[start:end])
        data = np.zeros((n_obs, self.data.shape[1]+2), dtype=np.float32)
        data[:,0] = self.data[start:end].index
        data[:,1] = -1
        data[:,2:] = np.array(self.data[start:end])

        data = data[1:] # generally cut off first datapoint

        n_block = n_obs // block
        obs_prop = np.zeros((n_block, data.shape[1]), dtype=np.float32)

        for i in xrange(n_block):
            ibeg = i*block
            iend = ibeg + block
            obs_prop[i] = data[ibeg:iend].mean(axis=0)

        header = '# Block average output for {}\n'.format(self.title) + self.header

        np.savetxt(filename, obs_prop, header=header, fmt="%.4f", comments="")

    def plot(self, ylim=None, start=0, block=1):
        pyplot.plot(self.data[:, 0], self.data[:, 1], label=self.name)
        #pyplot.plot(self.data[:, 0], self.data[:, 2], label=r'$beta P$')

        pyplot.xlabel(r'$kJ/mol$')

        if (ylim is not None):
            pyplot.ylim(ylim)

class PMFDataSet(DataSet):

    def __init__(self, root_filename,  kappa=1000, corr_len=1):
        super(PMFDataSet, self).__init__()

        data = np.loadtxt("{}/pullx.xvg".format(root_filename), comments=['@','#'])
        data[:,-1] = np.abs(data[:,-1])
        data_force = np.loadtxt("{}/pullf.xvg".format(root_filename), comments=['@','#'])

        self.inter_pos = None
        try:
            self.inter_pos = np.loadtxt("{}/slab_position.dat".format(root_filename))
            self.inter_pos /= 10.0 # in nm
        except:
            self.inter_pos = None

        dz_0 = data[-1, -1]
        force_0 = data_force[-1, -1]

        log.debug('Datareader {} reading input file {}'.format(self, "{}/pullx.xvg".format(root_filename)))
        
        # Ugh really hackish
        # NVT sim with 2 pull groups (slab, then solute)
        if data.shape[1] == 4:
            self.data = pandas.DataFrame(data[::corr_len, 1:], index=data[::corr_len, 0],
                                         columns=['r0', 'slabDZ', 'solDZ'])
        # NPT binding sim with 1 pull group (solute to its binding partner)
        if data.shape[1] == 3:
            self.data = pandas.DataFrame(data[::corr_len, 1:], index=data[::corr_len, 0],
                                         columns=['r0', 'solDZ'])
        self.title = root_filename
        self.kappa = kappa

        self.rstar = np.round(dz_0 + (force_0/self.kappa), decimals=3)


    def blockAvg(self, start, end=None, outfile=None):

        data = np.array(self.data[start:end]['solDZ'])
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
        pandas.DataFrame.rolling(self.data[start:end:10], window=block).plot()

        mean = self.getMean(start=start, end=end)
        line = pyplot.hlines(mean, start, self.shape[0])
        line.set_label('mean: {:.2f}'.format(mean))

        if (ylim is not None):
            pyplot.ylim(ylim)

    def getRange(self, start=0, end=None):
        rng = self.data[start:end].max() - self.data[start:end:10].min()

        return rng['solDZ']

    def max(self, start=0, end=None):
        return self.data[start:end].max()

    def min(self, start=0, end=None):
        return self.data[start:end].min()

    def getMean(self, start=0, bphi=1, end=None):
        dz = self.data[start:end]['solDZ']

        return Ntwid.mean()

    def getSecondMom(self, start=0, bphi=1, end=None):
        dz_sq = (self.data[start:end]['solDZ'])**2

        return dz_sq.mean()

    def getVar(self, start=0, bphi=1, end=None):
        dz = self.data[start:end]['solDZ']
        return dz.var()

    def getHist(self, start=0, nbins=50, end=None):
        return np.histogram(self.data[start:end]['solDZ'], bins=nbins)



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
    def loadPmf(cls, filename, kappa=1000.0, corr_len=1):
        ds = PMFDataSet(filename, kappa, corr_len)
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
    def plotHistAll(cls, start=0, end=None, nbins=50, idx='N', step=1, min_arr=False):
        total_array = np.array([])
        for title, dataset in cls.datasets.iteritems():
            total_array = np.append(total_array, dataset.data[start:end][idx])

        if min_arr:
            bins = np.arange(total_array.min(), total_array.max()+2*step, step)
        else:
            bins = np.arange(0, total_array.max()+2*step, step)
            
        pyplot.legend()
        counts, centers = np.histogram(total_array, bins=bins)
        centers = centers[:-1]
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
