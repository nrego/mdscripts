"""
Static data reader class for analysing phi time series data from simulations

nrego
"""
from __future__ import print_function, division; __metaclass__ = type
import numpy
import pandas
from matplotlib import pyplot
import logging
import re

log = logging.getLogger()
log.addHandler(logging.StreamHandler())


def normhistnd(hist, binbounds):
    '''Normalize the N-dimensional histogram ``hist`` with corresponding
    bin boundaries ``binbounds``.  Modifies ``hist`` in place and returns
    the normalization factor used.'''

    diffs = numpy.append(numpy.diff(binbounds), 0)

    assert diffs.shape == hist.shape
    normfac = (hist * diffs[0]).sum()

    hist /= normfac
    return normfac

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

    def __init__(self, filename):
        super(PhiDataSet, self).__init__()
        data = numpy.loadtxt(filename)
        log.debug('Datareader {} reading input file {}'.format(self, filename))
        self.data = pandas.DataFrame(data[:, 1:], index=data[:, 0],
                                     columns=['N', r'$\~N$'])
        self.title = filename

    def plot(self, start=0, ylim=None, block=1):
        pandas.rolling_mean(self.data[start:], window=block).plot()

        mean = self.getMean(start=start)
        line = pyplot.hlines(mean, start, self.shape[0])
        line.set_label('mean: {:.2f}'.format(mean))

        if (ylim is not None):
            pyplot.ylim(ylim)

    def getMean(self, start=0, bphi=1):
        #return self.data[start:].mean()[1]
        N = self.data[start:]['N']
        Ntwid = self.data[start:]['$\~N$']
        numer = (N*numpy.exp(bphi*(Ntwid-N)))
        denom = (numpy.exp(bphi*(Ntwid-N)))

        return numer.mean() / denom.mean()
        #return Ntwid.mean()

    def getVar(self, start=0, bphi=1):
        N_avg = self.getMean(start, bphi)

        N = self.data[start:]['N']
        Ntwid = self.data[start:]['$\~N$']
        numer = (N-N_avg)**2 * numpy.exp(bphi*(Ntwid-N))
        denom = numpy.exp(bphi*(Ntwid-N))

        return numer.mean() / denom.mean()

class XvgDataSet(DataSet):

    def __init__(self, title):
        super(XvgDataSet, self).__init__()

        floats = XvgDataSet._extractFloat(title)
        self.partner = None
        self.dual = False
        self.normfac = 0

        if len(floats) == 1:
            self.name = _titleString[0].format(floats[0])
            self.title = '{:.2f}'.format(floats[0])
        else:
            self.name = _titleString[1].format(floats[0], floats[1])
            self.title = '{:.2f}|{:.2f}'.format(floats[0], floats[1])
            self.dual = True
            #Opposite, for easier plotting - only match pairs once
            if floats[0] < floats[1]:
                self.partner = '{:.2f}|{:.2f}'.format(floats[1], floats[0])

        #self.title = title
        self.data = []

    def _rehashData(self):
        self.data = numpy.array(self.data)
        tmp_data = numpy.empty((self.data.shape[0], 3))
        if self.dual:
            negate = self.partner is not None
            conv = -0.4009 if negate else 0.4009
            tmp_data[:, :2] = self.data
            self.data = tmp_data
            # self.data[:, 1] modified in-place; self.data[:, 0] unchanged
            self.normfac = normhistnd(self.data[:, 1], self.data[:, 0])

            #self.data[:, 2] *= self.data[:, 1]
            #if negate:
            #    self.data[:, 0] *= -1
            self.data[:, 2] = numpy.log(self.data[:,1]) + (0.5*conv*self.data[:,0])

    def plot(self, ylim=None, start=0, block=1):
        pyplot.plot(self.data[:, 0], self.data[:, 1], label=self.name)
        #pyplot.plot(self.data[:, 0], self.data[:, 2], label=r'$beta P$')

        pyplot.xlabel(r'$kJ/mol$')

        if (ylim is not None):
            pyplot.ylim(ylim)

    @staticmethod
    def _extractFloat(string):
        return map(float, re.findall(r"[-+]?\d*\.\d+|\d+", string))


class DataReader:
    '''Global class for handling datasets, etc'''
    datasets = {}
    ts = None
    start = 0.0
    mean = -1
    phi = -1

    '''Load phiout.dat files'''
    @classmethod
    def loadPhi(cls, filename):
        ds = PhiDataSet(filename)
        return cls._addSet(ds)

    @classmethod
    def loadXVG(cls, filename):
        with open(filename) as xvg:
            titles = []
            curr_title_idx = -1
            curr_title = None
            curr_ds = None
            for i, line in enumerate(xvg):
                line.strip()
                if line.startswith("#") or len(line) == 0:
                    continue
                elif line.startswith("@ s"):
                    log.debug("str: {}".format(line.split('"')[1]))
                    title = line.split('"')[1]
                    titles.append(title)
                elif line.startswith("@\n"):
                    curr_title_idx += 1
                    curr_title = titles[curr_title_idx]
                    curr_ds = cls._addSet(XvgDataSet(curr_title))
                elif curr_ds:
                    curr_ds.data.append(map(float, line.split()))
                else:
                    continue

        for i, title in enumerate(cls.datasets):
            cls.datasets[title]._rehashData()

    @classmethod
    def _addSet(cls, ds):
        cls.datasets[ds.title] = ds
        return ds

    @classmethod
    def plot(cls, ylim=None, start=0, block=1):
        for title, dataset in cls.datasets.items():
            dataset.plot(ylim=ylim, start=start, block=block)

    @staticmethod
    def show():
        pyplot.legend()
        pyplot.show()


_titleString = [r'$N(dU/d\lambda | \lambda = {:.2f})$',
                r'$P(\Delta U(\lambda = {:.2f}) | \lambda = {:.2f}))$']

dr = DataReader
