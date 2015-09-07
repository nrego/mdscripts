import numpy
from matplotlib import pyplot
import argparse
import logging
from datareader import dr

import matplotlib as mpl

log = logging.getLogger('wham')

def parseRange(rangestr):
    spl = rangestr.split(",")
    return tuple([float(i) for i in spl])

# Generate S x M count histogram over Ntwid for all sims
# (M is number of bins)
def genHistMatrix(S, M, rng, start):

    histMat = numpy.empty((S,M))
    binbounds = numpy.empty((1,M+1))
    for i, ds in enumerate(dr.datasets.itervalues()):
        dataframe = ds.data[start:]['$\~N$'] # make this dynamic in future
        counts, binbounds = numpy.histogram(dataframe, bins=M, range=rng)
        histMat[i,:] = counts

    return histMat, binbounds

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

    

    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.INFO)
    if args.debug:
        log.setLevel(logging.DEBUG)

    infiles = args.input

    log.info("{} input files".format(len(infiles)))
    start = args.start

    conv = 1
    if args.T:
        conv /= (args.T * 0.008314)

    S = len(infiles) # Assume number of simulations from input

    nbins = args.nbins

    # N twid range
    data_range = loadRangeData(infiles, start)

    if args.range:
        data_range = parseRange(args.range)

    log.info('max, min: {}'.format(data_range))

    histMat, binbounds = genHistMatrix(S, nbins, data_range, start)

    log.info('Hist map shape: {}'.format(histMat.shape))
    log.debug('Bin bounds over range: {}'.format(binbounds))
