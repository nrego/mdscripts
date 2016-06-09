'''
Stitch together Ntwid histograms from phiout datafiles to construct
unbiased P(Ntwid) or P(N) using WHAM

nrego
Sept 2015
'''
import numpy
from scipy.optimize import fmin_bfgs
from matplotlib import pyplot
import argparse
import logging
from mdtools import dr
import uwham

import sys

import matplotlib as mpl

log = logging.getLogger()
log.addHandler(logging.StreamHandler())
Nfeval = 1

mpl.rcParams.update({'axes.labelsize': 30})
mpl.rcParams.update({'xtick.labelsize': 18})
mpl.rcParams.update({'ytick.labelsize': 18})
mpl.rcParams.update({'axes.titlesize': 36})

def normhistnd(hist, binbounds):
    '''Normalize the N-dimensional histogram ``hist`` with corresponding
    bin boundaries ``binbounds``.  Modifies ``hist`` in place and returns
    the normalization factor used.'''

    diffs = numpy.diff(binbounds)

    assert diffs.shape == hist.shape
    normfac = (hist * diffs[0]).sum()

    hist /= normfac
    return normfac

def parseRange(rangestr):
    spl = rangestr.split(",")
    return tuple([float(i) for i in spl])

# Generate S x M count histogram over Ntwid for all sims
# (M is number of bins)
def genDataMatrix(S, M, rng, start, end):

    dataMat = numpy.empty((S,M))
    binbounds = numpy.empty((1,M+1))
    for i, ds in enumerate(dr.datasets.itervalues()):
        dataframe = ds.data[start:end]['$\~N$'] # make this dynamic in future
        nsample, binbounds = numpy.histogram(dataframe, bins=M, range=rng)
        dataMat[i,:] = nsample

    return dataMat, binbounds

# Load list of infiles from start
# Return range over entire dataset (as tuple)
def loadRangeData(infiles, start, end):
    minval = float('inf')
    maxval = float('-inf')
    for f in infiles:
        ds = dr.loadPhi(f)

        # Min, max for this dataset
        tmpmax = ds.max(start=start, end=end)['$\~N$']
        tmpmin = ds.min(start=start, end=end)['$\~N$']

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

def genPdistBinless(all_data, u_nm, nsample_diag, weights, data_range, nbins):
    u_nm = numpy.array(u_nm)
    range_min, range_max = data_range
    nstep = float(range_max - range_min) / nbins
    #binbounds = numpy.arange(range_min, range_max+nstep, nstep)
    binbounds = numpy.linspace(range_min, range_max, nbins+1)

    nsample = numpy.diag(nsample_diag) * all_data.shape[0]

    weights_prod_nsample = nsample * weights

    pdist = numpy.zeros(nbins, dtype=numpy.float64)

    for n_idx in xrange(all_data.shape[0]):
        ntwid = all_data[n_idx]

        # which bin does ntwid fall into? A boolean array.
        bin_assign = (ntwid >= binbounds[:-1]) * (ntwid < binbounds[1:])

        denom = numpy.dot(weights_prod_nsample, u_nm[n_idx, :])

        pdist[bin_assign] += 1.0/denom

    return binbounds, pdist


# generate simulation weights from previously computed
#  (unbiased) probability distribution and bias matrix
def genWeights(prob, bias):
    weights = numpy.dot(bias, prob)

# For use with uwham - generate u_kln matrix of biased Ntwid values
def genU_kln(nsims, nsample, start, end, beta):
    maxnsample = nsample.max()
    u_kln = numpy.zeros((nsims, nsims, maxnsample))
    for i,ds_i in enumerate(dr.datasets.iteritems()):
        for j,ds_j in enumerate(dr.datasets.iteritems()):
            dataframe = numpy.array(ds_j[1].data[start:end]['$\~N$'])
            u_kln[i,j,:] = 0.5*ds_i[1].kappa*(dataframe-ds_i[1].Nstar)**2 + ds_i[1].phi*beta*dataframe

    #for k, l in numpy.ndindex(u_kln.shape[0:2]):
    #    u_kln[k, l, nsample[l]:] = numpy.NaN

    return u_kln

# Put all data points into N dim vector
def unpack_data(start, end=None):

    all_data = numpy.array([])
    nsample = numpy.array([])
    phivals = numpy.array([])

    for i, ds_item in enumerate(dr.datasets.iteritems()):
        ds_name, ds = ds_item
        dataframe = numpy.array(ds.data[start:end]['$\~N$'])
        nsample = numpy.append(nsample, dataframe.shape[0])
        all_data = numpy.append(all_data, dataframe)
        phivals = numpy.append(phivals, ds.phi)

    for val in nsample:
        if nsample.max() != val:
            log.info("NOTE: Data sets of unequal sizes")

    return all_data, numpy.matrix( numpy.diag(nsample/nsample.sum()) ), phivals

# U[i,j] is exp(-beta * Uj(n_i))
def genU_nm(all_data, nsims, beta, start, end=None):

    n_tot = all_data.shape[0]

    u_nm = numpy.zeros((n_tot, nsims))

    for i, (ds_name, ds) in enumerate(dr.datasets.iteritems()):
        u_nm[:, i] = numpy.exp( -beta*(0.5*ds.kappa*(all_data-ds.Nstar)**2 + ds.phi*all_data) )

    return numpy.matrix(u_nm)

# Log likelihood
def kappa(xweights, u_nm, nsample_diag, ones_m, ones_N, n_tot):

    logf = numpy.append(0, xweights)
    f = numpy.exp(-logf) # Partition functions relative to first window

    Q = numpy.dot(u_nm, numpy.diag(f))

    logLikelihood = (ones_N.transpose()/n_tot)*numpy.log(Q*nsample_diag*ones_m) + \
                    numpy.dot(numpy.diag(nsample_diag), logf)


    return float(logLikelihood)

def gradKappa(xweights, u_nm, nsample_diag, ones_m, ones_N, n_tot):

    logf = numpy.append(0, xweights)
    f = numpy.exp(-logf) # Partition functions relative to first window

    Q = numpy.dot(u_nm, numpy.diag(f))
    denom = (Q*nsample_diag).sum(axis=1)

    W = Q/denom

    grad = -nsample_diag*W.transpose()*(ones_N/n_tot) + nsample_diag*ones_m

    return ( numpy.array(grad[1:]) ).reshape(len(grad)-1 )

def callbackF(xweights):
    global Nfeval
    #log.info('Iteration {}'.format(Nfeval))
    log.info('\rIteration {}\r'.format(Nfeval))
    sys.stdout.flush()
    #log.info('.')

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

def logP(N, fk_inv, nsample, phivals, beta):

    return -numpy.log( (nsample*fk_inv*numpy.exp(-beta*N*phivals)).sum() )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run WHAM on collection of INDUS output datasets")

    parser.add_argument('input', metavar='INPUT', type=str, nargs='+',
                        help='Input file names (presumably in a sensible order)')
    parser.add_argument('-b', '--start', type=int, default=0,
                        help='first timepoint (in ps)')
    parser.add_argument('-e', '--end', type=int, default=None,
                        help='last timepoint (in ps) - default is last available time point')
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
    parser.add_argument('--plotLogP', action='store_true',
                        help='Plot resulting log probability (log(P))')
    parser.add_argument('--uwham', action='store_true',
                        help='perform UWHAM analysis (default: False)')
    parser.add_argument('--mywham', action='store_true', 
                        help='perform WHAM analysis with my implementation (improved convergence)')

    parser.add_argument('--logweights', default=None,
                        help='Input file for WHAM weights, if previously calculated')

    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.INFO)
    if args.debug:
        log.setLevel(logging.DEBUG)

    infiles = args.input

    log.info("{} input files".format(len(infiles)))
    start = args.start
    end = args.end

    beta = 1
    if args.T:
        beta /= (args.T * 8.314462e-3)

    nsims = len(infiles) # Assume number of simulations from input

    nbins = args.nbins

    # N twid range
    data_range = loadRangeData(infiles, start, end)

    if args.range:
        data_range = parseRange(args.range)

    log.info('max, min: {}'.format(data_range))

    dataMat, binbounds = genDataMatrix(nsims, nbins, data_range, start, end)

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


    

    ## UWHAM analysis
    if args.uwham:
        u_kln = genU_kln(nsims, nsample.max(), start, end, beta)
        log.info('u_kln shape: {}'.format(u_kln.shape))
        log.debug('u_kln[1,1,:]: {}'.format(u_kln[1,1,:]))
        results = uwham.UWHAM(u_kln, nsample)
        numpy.savetxt('uwham_results.dat', results.f_k, fmt='%3.3f')
        logweights = results.f_k
        weights = numpy.exp(logweights)

    # My WHAM implementation
    elif args.mywham:

        all_data, nsample_diag, phivals = unpack_data(start, end)
        u_nm = genU_nm(all_data, nsims, beta, start, end)

        xweights = numpy.zeros(nsims-1)
        ones_m = numpy.matrix(numpy.ones(nsims)).transpose()
        ones_N = numpy.matrix(numpy.ones(all_data.shape[0])).transpose()

        n_tot = len(all_data)
        myargs = (u_nm, nsample_diag, ones_m, ones_N, n_tot)

        log.info("Beginning optimization procedure...")
        res = fmin_bfgs(kappa, xweights, fprime=gradKappa, args=myargs)
        logweights = numpy.append(0, -res)
        weights = numpy.exp(logweights)
        numpy.savetxt('logweights.dat', logweights, fmt='%3.3f')

    elif args.logweights is not None:
        try:
            logweights = numpy.loadtxt(args.logweights)
        except IOError:
            print "Error loading weights file '{}'".format(args.logweights)
        all_data, nsample_diag, phivals = unpack_data(start, end)
        u_nm = genU_nm(all_data, nsims, beta, start, end)
        weights = numpy.exp(logweights)



    #probDist = genPdist(dataMat, weights, nsample, numer, biasMat)
    binbounds, probDist = genPdistBinless(all_data, u_nm, nsample_diag, weights, data_range, nbins)

    log.debug('pdist (pre-normalization): {}'.format(probDist))
    normfac = normhistnd(probDist, binbounds)
    log.info('norm fac: {}'.format(normfac))

    log.info('pdist shape:{}'.format(probDist.shape))
    log.debug('pdist: {}'.format(probDist))

    if args.plotPdist:
        pyplot.plot(bincntrs, probDist)
        pyplot.show()

    elif args.plotE:
        pyplot.plot(bincntrs, -numpy.log(probDist))
        pyplot.show()

    elif args.plotLogP:
        pyplot.plot(bincntrs, numpy.log(probDist))
        pyplot.show()

    log.info('shape of stacked array: {}'.format(numpy.column_stack((bincntrs, probDist)).shape))
    numpy.savetxt('Pn.dat', numpy.column_stack((bincntrs, probDist)),
                  fmt='%3.3f %1.3e')
    numpy.savetxt('logPn.dat', numpy.column_stack((bincntrs, numpy.log(probDist))),
                  fmt='%3.3f %3.3f')
    numpy.savetxt('binbounds', binbounds, fmt='%3.3f')

    log.info('nsims: {}'.format(nsims))
    log.info('N_k shape: {}'.format(nsample.shape))
