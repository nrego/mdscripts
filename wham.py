import numpy
from matplotlib import pyplot
import argparse
import logging
from datareader import dr

import matplotlib as mpl



def genHistMatrix(S, M):




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run WHAM on collection of INDUS output datasets")

    parser.add_argument('input', metavar='INPUT', type=str, nargs='+',
                        help='Input file names')
    parser.add_argument('-b', '--start', type=int, default=0,
                        help='first timepoint (in ps)')
    parser.add_argument('--debug', action='store_true',
                        help='print debugging info')
    parser.add_argument('-T', metavar='TEMP', type=float,
                        help='convert Phi values to kT, for TEMP (K)')
    parser.add_argument('-nbins', metavar='NBINS', type=int, default=50,
                        help='Number of histogram bins')

    log = logging.getLogger()
    log.addHandler(logging.StreamHandler())

    args = parser.parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)

    infiles = args.input

    log.debug("{} input files".format(len(infiles)))
    start = args.start

    conv = 1
    if args.T:
        conv /= (args.T * 0.008314)

    S = args.S or len(infiles) # Number of simulations to WHAM - assumed to be from input files

    histMat = getHistMatrix(S, nbins)