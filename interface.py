#Instantaneous interface

import numpy
import argparse
import logging

import MDAnalysis
from MDAnalysis.coordinates.xdrfile.libxdrfile2 import read_xtc_natoms, xdrfile_open

import scipy.spatial
from skimage import measure

from phi import phi



if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Instantaneous interface analysis code. Either generate isosurface or analyze results")

    parser.add_argument('-c', '--grofile', metavar='INPUT', type=str, required=True,
                        help='Input structure file')
    parser.add_argument('-f', '--trajfile', metavar='XTC', type=str, required=True,
                        help='Input XTC trajectory file')
    parser.add_argument('-d', '--cutoff', type=float, default=7.0,
                        help='Cutoff distance for coarse grained gaussian, in Angstroms (default: 7 A)')
    parser.add_argument('-b', '--start', type=int, default=0,
                        help='First timepoint (in ps)')
    parser.add_argument('-o', '--outtraj', default='traj_II.xtc',
                        help='Output trajectory file to write instantaneous interface evolution')


    log = logging.getLogger('interface')


    args = parser.parse_args()


    cutoff = args.cutoff
    # phi look up hash input increment (0.1 A or 0.01 nm)
    dl = 0.1

    #sigma_w = 0.24
    #sigma_p = 0.24

    # Phi hash for distances up to cutoff
    phi_hash = phi(numpy.arange(0, cutoff+dl,dl), sigma, cutoff)

    u = MDAnalysis.Universe(args.grofile, args.trajfile)

    # Start frame, in trajectory units
    startframe = int(args.start / u.trajectory.dt)
    lastframe = int(u.trajectory.n_frames / u.trajectory.dt)
    nframes = lastframe - startframe

    prot_heavies = u.select_atoms("not (name H* or resname SOL) and not (name CL or name NA)")
    water_ow = u.select_atoms("name OW")
    ions = u.select_atoms("name CL or name NA")

    # Hard coded for now - obviously must predict
    rho_water_bulk = 0.0320
    rho_prot_bulk = 0.0410
    sigma = 2.4

    grid_dl = 1
    natoms = u.coord.n_atoms

    # Numpy 6d array of xyz dimensions - I assume last three dimensions are axis angles?
    # should modify in future to accomodate non cubic box
    #   In angstroms
    box = u.dimensions


    # Set up marching cube stuff - grids in Angstroms
    #  ngrids are grid dimensions of discretized space at resolution ngrids[i] in each dimension
    ngrids = box[:3].astype(int)+1
    dgrid = box[:3]/(ngrids-1)

    npts = ngrids.prod()
    ninc = cutoff/grid_dl
    coord_x = numpy.arange(0,ngrids[0],dgrid[0])
    coord_y = numpy.arange(0,ngrids[1],dgrid[1])
    coord_z = numpy.arange(0,ngrids[2],dgrid[2])
    xpts, ypts, zpts = numpy.meshgrid(coord_x,coord_y,coord_z)

    # Todo: must account for periodicity - i.e. all grid points within
    #   cutoff distance of box edge have to be reflected on the opposite side
    gridpts = numpy.array(zip(xpts.ravel(),ypts.ravel(),zpts.ravel()))
    grididx = gridpts / dgrid

    rho_water = numpy.zeros((nframes, npts), dtype=numpy.float32)
    rho_prot = numpy.zeros((nframes, npts), dtype=numpy.float32)
    rho = numpy.zeros((nframes, npts), dtype=numpy.float32)

    # KD tree for nearest neighbor search
    tree = scipy.spatial.KDTree(gridpts)

    for i in xrange(startframe, lastframe):
        print "Frame: {}".format(i)

        u.trajectory[i]

        for atom in prot_heavies:

            pos = atom.position
            neighboridx = numpy.array(tree.query_ball_point(pos, cutoff))
            neighborpts = gridpts[neighboridx]

            # Distance array between atom and neighbor grid points
            distarr = scipy.spatial.distance.cdist(pos.reshape(1,3), neighborpts).reshape(neighboridx.shape)

            phivals = phi(distarr, sigma, cutoff)

            rho_prot[i-startframe, neighboridx] += phivals

        for atom in water_ow:
            pos = atom.position
            neighboridx = numpy.array(tree.query_ball_point(pos, cutoff))
            neighborpts = gridpts[neighboridx]

            # Distance array between atom and neighbor grid points
            distarr = scipy.spatial.distance.cdist(pos.reshape(1,3), neighborpts).reshape(neighboridx.shape)

            phivals = phi(distarr, sigma, cutoff)

            rho_water[i-startframe, neighboridx] += phivals

        rho[i-startframe,:] = rho_prot[i-startframe,:]/rho_prot_bulk + rho_water[i-startframe,:]/rho_water_bulk
