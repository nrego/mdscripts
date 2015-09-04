'''
Stitch together Ntwid histograms from phiout datafiles to construct
unbiased P(Ntwid) using WHAM

nrego
Sept 2015 
'''
import numpy
from matplotlib import pyplot
import argparse
import logging
from datareader import dr

import matplotlib as mpl

log = logging.getLogger('wham')

def normhistnd(hist, binbounds):
    '''Normalize the N-dimensional histogram ``hist`` with corresponding
    bin boundaries ``binbounds``.  Modifies ``hist`` in place and returns
    the normalization factor used.'''

    diffs = numpy.append(numpy.diff(binbounds), 0)

    assert diffs.shape == hist.shape
    normfac = (hist * diffs[0]).sum()

    hist /= normfac
    return normfac

def parseRange(rangestr):
    spl = rangestr.split(",")
    return tuple([float(i) for i in spl])

# Generate S x M count histogram over Ntwid for all sims
# (M is number of bins)
def genDataMatrix(S, M, rng, start):

    dataMat = numpy.empty((S,M))
    binbounds = numpy.empty((1,M+1))
    for i, ds in enumerate(dr.datasets.itervalues()):
        dataframe = ds.data[start:]['$\~N$'] # make this dynamic in future
        nsample, binbounds = numpy.histogram(dataframe, bins=M, range=rng)
        dataMat[i,:] = nsample

    return dataMat, binbounds

# Load list of infiles from start
# Return range over entire dataset (as tuple)
def loadRangeData(infiles, start):
    minval = float('inf')
    maxval = float('-inf')
    for f in infiles:
        ds = dr.loadPhi(f)

        # Min, max for this dataset
        tmpmax = ds.max(start=start)['$\~N$']
        tmpmin = ds.min(start=start)['$\~N$']

        minval = tmpmin if minval>tmpmin else minval
        maxval = tmpmax if maxval<tmpmax else maxval


    return minval, maxval    

# generate P histogram (shape: nbins) from weights (shape: nsims),
#   bias histogram N (nsims X nbins matrix), and bias (nsims X nbins matrix)
# **Hopefully** this vectorized implementation will be reasonably quick
#   since it relies on optimized numpy matrix manipulation routines
def genPdist(data, weights, nsample, numer, bias):

    nsims, nbins = data.shape

    probHist = numpy.empty(nbins)

    denom = numpy.dot(weights*nsample, bias)
    probHist = numer/denom


    return probHist

# generate simulation weights from previously computed 
#  (unbiased) probability distribution and bias matrix
def genWeights(prob, bias):
    weights = numpy.dot(bias, prob)

# Generate nsims x nbins bias matrix
def genBias(bincntrs, beta):
    nsims = len(dr.datasets)
    nbins = len(bincntrs)

    biasMat = numpy.empty((nsims, nbins))

    for i, ds in enumerate(dr.datasets.itervalues()):
        kappa = ds.kappa
        Nstar = ds.Nstar
        phi = ds.phi

        biasMat[i, :] = 0.5*kappa*(bincntrs-Nstar)**2 + phi*bincntrs

    biasMat = numpy.exp(-beta*biasMat)

    return biasMat

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run WHAM on collection of INDUS output datasets")

    parser.add_argument('input', metavar='INPUT', type=str, nargs='+',
                        help='Input file names (presumably in a sensible order)')
    parser.add_argument('-b', '--start', type=int, default=0,
                        help='first timepoint (in ps)')
    parser.add_argument('--debug', action='store_true',
                        help='print debugging info')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='be verbose')
    parser.add_argument('-T', metavar='TEMP', type=float,
                        help='convert Phi values to kT, for TEMP (K)')
    parser.add_argument('-nbins', metavar='NBINS', type=int, default=50,
                        help='Number of histogram bins (over range if specified, or from \
                            the minium to maximum of the input[start:]')
    parser.add_argument('--range', type=str,
                        help="Specify custom data range (as 'minval,maxval') for \
                        histogram binning (default: Full data range)")
    parser.add_argument('--maxiter', type=int, metavar='ITER', default=1000,
                        help='Maximum number of iterations to evaluate WHAM functions')

    

    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.INFO)
    if args.debug:
        log.setLevel(logging.DEBUG)

    infiles = args.input

    log.info("{} input files".format(len(infiles)))
    start = args.start

    beta = 1
    if args.T:
        beta /= (args.T * 8.314462e-3)

    nsims = len(infiles) # Assume number of simulations from input

    nbins = args.nbins

    # N twid range
    data_range = loadRangeData(infiles, start)

    if args.range:
        data_range = parseRange(args.range)

    log.info('max, min: {}'.format(data_range))

    dataMat, binbounds = genDataMatrix(nsims, nbins, data_range, start)

    log.info('Hist map shape: {}'.format(dataMat.shape))
    log.debug('Bin bounds over range: {}'.format(binbounds))

    bincntrs = (binbounds[1:]+binbounds[:-1])/2.0

    biasMat = genBias(bincntrs, beta)

    weights = numpy.ones(nsims)
    nsample = dataMat.sum(1) # Number of data points for each simulation 
    numer = dataMat.sum(0) # Sum over all simulations of nsample in each bin

    for i in xrange(args.maxiter):
        probDist = genPdist(dataMat, weights, nsample, numer, biasMat)
        weights = 1/numpy.dot(biasMat, probDist)

    normhistnd(probDist, binbounds[:-1])

    pyplot.plot(bincntrs, -numpy.log(probDist))
    pyplot.show()

    numpy.savetxt('Pn.dat', probDist)


