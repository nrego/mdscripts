#Instantaneous interface

import numpy
import math
import argparse
import logging

import MDAnalysis
from MDAnalysis.coordinates.xdrfile.libxdrfile2 import read_xtc_natoms, xdrfile.open

# Caluclates value of coarse-grained gaussian for point at r
#   sigma and cutoff in nm
@numpy.vectorize
def phi(r, sigma=0.24, cutoff=0.7):

    if (abs(r) > cutoff):
        return 0.0

    else:
        phic = math.exp(-cutoff**2/(2*sigma**2))
        pref = 1 / ( (2*math.pi)**(0.5) * sigma * math.erf(cutoff / (2**0.5 * sigma)) - 2*cutoff*phic )

        return pref * ( math.exp(-r**2/(2*sigma**2)) ) - phic



if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Instantaneous interface analysis code. Either generate isosurface or analyze results")

    parser.add_argument('-c', '--grofile', metavar='INPUT', type=str, required=True,
                        help='Input structure file')
    parser.add_argument('-f', '--trajfile', metavar='XTC', type=str, required=True,
                        help='Input XTC trajectory file')
    parser.add_argument('-d', '--cutoff', type=float, default=0.7,
                        help='Cutoff distance for coarse grained gaussian (default: 0.7)')
    parser.add_argument('-b', '--start', type=int, default=0,
                        help='First timepoint (in ps)')
    parser.add_argument('-o', '--outtraj', default='traj_II.xtc',
                        help='Output trajectory file to write instantaneous interface evolution')


    log = logging.getLogger('interface')


    args = parser.parse_args()


    cutoff = args.cutoff
    dl = 0.01

    #sigma_w = 0.24
    #sigma_p = 0.24

    # Phi hash for distances up to cutoff
    phi_hash = phi(numpy.arange(0,cutoff,dl))

    u = MDAnalysis.universe(args.grofile, args.trajfile)

    # Start frame, in trajectory units
    startframe = args.start / u.trajectory.ts

    protHeavies = u.selectAtoms("not (name H* or resname SOL) and not (name CL or name NA)")
    waterHeavies = u.selectAtoms("name OW")
    ions = u.selectAtoms("name CL or name CL")

    # Hard coded for now - obviously must predict
    rho_water_bulk = 32.0
    rho_prot_bulk = 41.0

    grid_dl = 1
    natoms = u.coord.numatoms

    # Numpy 6d array of xyz dimensions - I assume last three dimensions are axis angles?
    # should modify in future to accomodate non cubic box
    #   In angstroms
    box = u.dimensions


    # Set up marching cube stuff - grids in Angstroms
    ngrids = box[:3].astype(int)
    dgrid = box[:3]/ngrids

    ncubes = ngrids.prod()
    ninc = cutoff/grid_dl


    for i in xrange(startframe, u.trajectory.numframes):

        u.trajectory.frame[i]

        