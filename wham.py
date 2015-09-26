'''
Stitch together Ntwid histograms from phiout datafiles to construct
unbiased P(Ntwid) using WHAM

nrego
Sept 2015
'''
import numpy
import scipy.optimize
from matplotlib import pyplot
import argparse
import logging
from datareader import dr
import uwham
from pymbar import MBAR

import matplotlib as mpl

log = logging.getLogger('wham')
Nfeval = 1

mpl.rcParams.update({'axes.labelsize': 30})
mpl.rcParams.update({'xtick.labelsize': 18})
mpl.rcParams.update({'ytick.labelsize': 18})
mpl.rcParams.update({'axes.titlesize': 36})

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
# This returns an array of 'nan' if (I assume) we drop below machine precision -
#    TODO: Implement a graceful check to avoid this
def genPdist(data, weights, nsample, numer, bias):

    nsims, nbins = data.shape

    denom = numpy.dot(weights*nsample, bias)

    probHist = numer/denom


    return probHist

# generate simulation weights from previously computed
#  (unbiased) probability distribution and bias matrix
def genWeights(prob, bias):
    weights = numpy.dot(bias, prob)

# For use with uwham - generate u_kln matrix of biased Ntwid values
def genU_kln(nsims, nsample, start, beta):
    maxnsample = nsample.max()
    u_kln = numpy.zeros((nsims, nsims, maxnsample))
    for i,ds_i in enumerate(dr.datasets.iteritems()):
        for j,ds_j in enumerate(dr.datasets.iteritems()):
            dataframe = numpy.array(ds_j[1].data[start:]['$\~N$'])
            u_kln[i,j,:] = ds_i[1].phi*beta*dataframe

    #for k, l in numpy.ndindex(u_kln.shape[0:2]):
    #    u_kln[k, l, nsample[l]:] = numpy.NaN

    return u_kln

# Log likelihood
def kappa(xweights, nsample, u_kln):
    logweights = numpy.zeros(xweights.shape[0]+1)
    logweights[1:] = xweights
    weights = numpy.exp(logweights)
    mat = numpy.zeros(u_kln.shape[1:])
    for j in xrange(u_kln.shape[1]):
        mat[j] = numpy.log(numpy.dot((weights*nsample), numpy.exp(-u_kln[:,j,:])))

    logLikelihood = numpy.nansum(mat) - numpy.dot(nsample, logweights)

    return logLikelihood

def gradKappa(xweights, nsample, u_kln):
    logweights = numpy.zeros(xweights.shape[0]+1)
    logweights[1:] = xweights
    weights = numpy.exp(logweights)
    grad = numpy.zeros(logweights.shape)
    mat = numpy.empty(u_kln.shape[1]*u_kln.shape[2])
    npts = u_kln.shape[2]
    for j in xrange(u_kln.shape[1]):
        mat[j*npts:(j+1)*npts] = numpy.dot((weights*nsample), numpy.exp(-u_kln[:,j,:]))

    inv_mat = numpy.power(mat, -1)
    for i in xrange(grad.shape[0]):
        grad[i] = numpy.dot(inv_mat*nsample[i]*weights[i],
                             numpy.exp(-numpy.ravel(u_kln[i]))) - nsample[i]

    return grad[1:]

def callbackF(xweights):
    global Nfeval
    #log.info('Iteration {}'.format(Nfeval))
    print 'Iteration {}'.format(Nfeval)
    Nfeval += 1

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
    parser.add_argument('--nbins', metavar='NBINS', type=int, default=50,
                        help='Number of histogram bins (over range if specified, or from \
                            the minium to maximum of the input[start:]')
    parser.add_argument('--range', type=str,
                        help="Specify custom data range (as 'minval,maxval') for \
                        histogram binning (default: Full data range)")
    parser.add_argument('--maxiter', type=int, metavar='ITER', default=1000,
                        help='Maximum number of iterations to evaluate WHAM functions')
    parser.add_argument('--plotPdist', action='store_true',
                        help='Plot resulting probability distribution')
    parser.add_argument('--plotE', action='store_true',
                        help='Plot resulting (free) energy distribution (-log(P))')
    parser.add_argument('--uwham', action='store_true',
                        help='perform UWHAM analysis (default: False)')



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
    binwidths = (binbounds[1:]-binbounds[:-1])/2.0
    log.debug('Bin centers: {}'.format(binbounds))
    biasMat = genBias(bincntrs, beta)

    weights = numpy.ones(nsims, dtype=numpy.float64)
    logweights = numpy.log(weights)
    nsample = dataMat.sum(1) # Number of data points for each simulation

    numer = dataMat.sum(0) # Sum over all simulations of nsample in each bin

    # Track the weights for each simulation over time
    convergenceMat = numpy.zeros((100, nsims), numpy.float64)
    #printinter = args.maxiter/100
    '''
    for i in xrange(args.maxiter):
        printinfo = (i%printinter == 0)
        probDist = genPdist(dataMat, weights, nsample, numer, biasMat)
        weights = 1/numpy.dot(biasMat, probDist)
        if printinfo:
            convergenceMat[i/printinter,:] = numpy.log(weights/weights[0])
            log.info('\rIter {}'.format(i))
            log.debug('probDist: {}'.format(probDist))
            log.debug('weights: {}'.format(weights))
    '''
    u_kln = genU_kln(nsims, nsample.max(), start, beta)

    ## UWHAM analysis
    if args.uwham:
        log.info('u_kln shape: {}'.format(u_kln.shape))
        log.debug('u_kln[1,1,:]: {}'.format(u_kln[1,1,:]))
        results = uwham.UWHAM(u_kln, nsample)
        numpy.savetxt('uwham_results.dat', results.f_k, fmt='%3.3f')
        logweights = results.f_k
        weights = numpy.exp(logweights)

    else:
        log.info('Starting optimization...')
        logweights[1:] = scipy.optimize.fmin_bfgs(f=kappa, x0=logweights[1:], fprime=gradKappa,
                                              args=(nsample, u_kln), callback=callbackF)
        weights = numpy.exp(logweights)
        log.info('Free energies for each umbrella: {}'.format(logweights))

    probDist = genPdist(dataMat, weights, nsample, numer, biasMat)

    log.debug('pdist (pre-normalization): {}'.format(probDist))
    normfac = normhistnd(probDist, binbounds[:-1])
    log.info('norm fac: {}'.format(normfac))

    log.info('pdist shape:{}'.format(probDist.shape))
    log.debug('pdist: {}'.format(probDist))

    if args.plotPdist:
        pyplot.plot(bincntrs, probDist)
        pyplot.show()

    elif args.plotE:
        pyplot.plot(bincntrs, -numpy.log(probDist))
        pyplot.show()

    log.info('shape of stacked array: {}'.format(numpy.column_stack((bincntrs, probDist)).shape))
    numpy.savetxt('Pn.dat', numpy.column_stack((bincntrs, probDist)),
                  fmt='%3.3f %1.3e')
    numpy.savetxt('logPn.dat', numpy.column_stack((bincntrs, -numpy.log(probDist))),
                  fmt='%3.3f %3.3f')
    numpy.savetxt('convergence.dat', convergenceMat, fmt='%3.3f')

    log.info('nsims: {}'.format(nsims))
    log.info('N_k shape: {}'.format(nsample.shape))
