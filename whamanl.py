import numpy
from scipy.optimize import fmin_bfgs
from matplotlib import pyplot
import argparse
import logging
from datareader import dr
import uwham


import sys

import matplotlib as mpl

log = logging.getLogger('whamanl')
Nfeval = 1

mpl.rcParams.update({'axes.labelsize': 30})
mpl.rcParams.update({'xtick.labelsize': 18})
mpl.rcParams.update({'ytick.labelsize': 18})
mpl.rcParams.update({'axes.titlesize': 36})


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run WHAM analysis")

    parser.add_argument('input', metavar='INPUT', type=str, nargs='+',
                        help='Input file names (presumably in a sensible order)')
    parser.add_argument('-b', '--start', type=int, default=0,
                        help='first timepoint (in ps)')
    parser.add_argument('-e', '--end', type=int, default=None,
                        help='last timepoint (in ps) - default is last available time point')
    parser.add_argument('--logweights', default='logweights.dat', 
                        help='Input file for WHAM logweights, if previously calculated')
    parser.add_argument('--pdist', default='Pn.dat', 
                        help='Name of calculated pdist file (default: Pn.dat)')
    parser.add_argument('--debug', action='store_true',
                        help='print debugging info')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='be verbose')
    parser.add_argument('-T', metavar='TEMP', type=float,
                        help='convert Phi values to kT, for TEMP (K)')
    parser.add_argument('--plotPdist', action='store_true',
                        help='Plot resulting probability distribution')
    parser.add_argument('--plotE', action='store_true',
                        help='Plot resulting (free) energy distribution (-log(P))')
    parser.add_argument('--plotLogP', action='store_true',
                        help='Plot resulting log probability (log(P))')

    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.INFO)
    if args.debug:
        log.setLevel(logging.DEBUG)

    log.info("Loading input files")

    for infile in args.input:
        dr.loadPhi(infile)

    log.info("   ...Done")

    start = args.start
    end = args.end 

    logweights = numpy.loadtxt(args.logweights)
    weights = numpy.exp(logweights)

    probDist = numpy.loadtxt(args.pdist)
    logP = numpy.exp(probDist)

    binctrs = probDist[:, 0]
    binlen = numpy.diff(binctrs)[0] # Assume equal sized bins

    # Contains calculated weights (i.e. relative partition coef for each sim)
    #   And shannon entropy over consensus and observed distributions
    outarr = numpy.zeros((len(args.input), 3))

    beta = 1
    if args.T:
        beta /= (args.T * 8.314462e-3)

    for i, ds_item in enumerate(dr.datasets.iteritems()):
        ds_name, ds = ds_item
        bias = numpy.exp(-beta*(0.5*ds.kappa*(binctrs-ds.Nstar)**2 + ds.phi*binctrs))
        calcWeight = (probDist[:, 1] * bias * binlen).sum()
        log.info("Calculated weight for nstar={}, phi={}:  f={}".format(ds.Nstar, ds.phi, -numpy.log(calcWeight)))
        # Consensus histogram
        biasDist = (probDist[:, 1] * bias * binlen) / calcWeight

        range_max = (binctrs + (numpy.diff(binctrs)/2.0)[0])[-1]
        range_min = (binctrs - (numpy.diff(binctrs)/2.0)[0])[0]

        data_arr = numpy.array(ds.data[start:end]['$\~N$'])
        obs_hist, bb = numpy.histogram(data_arr, range=(range_min, range_max), bins=len(binctrs), normed=True)

        comp_arr = numpy.zeros((obs_hist.shape[0], 3))
        comp_arr[:, 0] = (bb[:-1] + numpy.diff(bb)/2.0)
        comp_arr[:, 1] = obs_hist[:]
        comp_arr[:, 2] = biasDist[:]

        numpy.savetxt('nstar-{:05g}_phi-{:05g}_consensus.dat'.format(ds.Nstar, ds.phi*1000), comp_arr)

        shannonEntropy = numpy.nansum(obs_hist*numpy.log(obs_hist/biasDist))
        log.info("  Shannon Entropy: {}".format(shannonEntropy))

        outarr[i, 0] = beta*ds.phi
        outarr[i, 1] = -numpy.log(calcWeight)
        outarr[i, 2] = shannonEntropy

    numpy.savetxt('wham_anl.dat', outarr, fmt='%3.3f')

    if args.plotPdist:
        pyplot.plot(probDist[:,0], probDist[:,1])
        pyplot.show()

    elif args.plotE:
        pyplot.plot(probDist[:,0], -numpy.log(probDist[:,1]))
        pyplot.show()

    elif args.plotLogP:
        pyplot.plot(probDist[:,0], numpy.log(probDist[:,1]))
        pyplot.show()