#Instantaneous interface

import numpy
import math
import argparse
import logging

import MDAnalysis

# Caluclates value of coarse-grained gaussian for point at r
#   sigma and cutoff in nm
def phi(r, sigma=0.24, cutoff=0.7):

	if (abs(r) > cutoff):
		return 0.0

	else:
		phic = math.exp(-cutoff**2/(2*sigma**2))
		pref = 1 / ( (2*math.pi)**(0.5) * sigma * math.erf(cutoff / (2**0.5 * sigma)) - 2*cutoff*phic )

		return pref * ( math.exp(-r**2/(2*sigma**2)) ) - phic



if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Instantaneous interface analysis code. Either generate isosurface or analyze results")

    parser.add_argument('input', metavar='INPUT', type=str, nargs='+',
                        help='Input file names')
    parser.add_argument('-d', '--cutoff', type=float, default=0.7,
    					help='Cutoff distance for coarse grained gaussian (default: 0.7)')
    parser.add_argument()


    log = logging.getLogger('interface')


    args = parser.parse_args()

    cutoff = args.cutoff

    #sigma_w = 0.24
    #sigma_p = 0.24

    npts = int(cutoff/0.01)

 