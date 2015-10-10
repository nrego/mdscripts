#Instantaneous interface
# nrego sept 2015

from __future__ import division; __metaclass__ = type
import sys
import numpy
import argparse
import logging

import MDAnalysis
from MDAnalysis.coordinates.xdrfile.libxdrfile2 import read_xtc_natoms, xdrfile_open

import scipy.spatial
from skimage import measure

from utils import phi
from mdtools import ParallelTool

class Interface(ParallelTool):
    prog='w_assign'
    description = '''\
Perform instantaneous interface analysis on simulation data. Requires 
GROFILE and TRAJECTORY (XTC or TRR). Conduct Instantaneous interface 
analysis over specified trajectory range.

This tool supports parallelization (see options below)


-----------------------------------------------------------------------------
Command-line options
-----------------------------------------------------------------------------
'''
    
    def __init__(self):
        super(Interface,self).__init__()
        
        # Parallel processing by default (this is not actually necessary, but it is
        # informative!)
        self.wm_env.default_work_manager = self.wm_env.default_parallel_work_manager

        self.univ = None

        self.cutoff = None
        self.sigma = None
        self.rho_prot_bulk = None
        self.rho_water_bulk = None

        self.start_frame = None
        self.last_frame = None

        self.output_filename = None

    @property
    def n_frames(self):
        return self.last_frame - self.start_frame
        
    
    def add_args(self, parser):
        
        sgroup = parser.add_argument_group('Instantaneous interface options')
        sgroup.add_argument('-c', '--grofile', metavar='INPUT', type=str, required=True,
                            help='Input structure file')
        sgroup.add_argument('-f', '--trajfile', metavar='XTC', type=str, required=True,
                            help='Input XTC trajectory file')
        sgroup.add_argument('-d', '--cutoff', type=float, default=7.0,
                            help='Cutoff distance for coarse grained gaussian, in Angstroms (default: 7 A)')
        sgroup.add_argument('-b', '--start', type=int, default=0,
                        help='First timepoint (in ps)')
        sgroup.add_argument('--rhoprot', default=40, type=float,
                        help='Estimated protein density (heavy atoms per nm3)')

        agroup = parser.add_argument_group('other options')
        agroup.add_argument('-o', '--outfile', default='interface.dx',
                        help='Output file to write instantaneous interface')


    def process_args(self, args):

        try:
            self.univ = u = MDAnalysis.Universe(args.grofile, args.trajfile)
        except:
            #print "Error processing input files: {} and {}".format(args.grofile, args.trajfile)
            sys.exit()

        self.cutoff = args.cutoff
        self.sigma = 2.4
        self.rho_water_bulk = 0.0330
        self.rho_prot_bulk = args.rhoprot / 1000.0

        if (args.start > (u.trajectory.n_frames * u.trajectory.dt)):
            raise ValueError("Error: provided start time ({} ps) is greater than total time ({} ps)"
                             .format(args.start, (u.trajectory.n_frames * u.trajectory.dt)))

        self.start_frame = int(args.start / u.trajectory.dt)
        self.last_frame = u.trajectory.n_frames

        self.output_filename = args.outfile

    def calc_rho(self, npts, gridpts, prot_heavies, water_ow):

        cutoff_sq = self.cutoff**2
        sigma_sq = self.sigma**2

        rho_water = numpy.zeros((self.n_frames, npts), dtype=numpy.float32)
        rho_prot = numpy.zeros((self.n_frames, npts), dtype=numpy.float32)
        rho = numpy.zeros((self.n_frames, npts), dtype=numpy.float32)

        # KD tree for nearest neighbor search
        tree = scipy.spatial.cKDTree(gridpts)

        for i in xrange(self.start_frame, self.last_frame):
            print "Frame: {}".format(i)

            self.univ.trajectory[i]

            for atom in prot_heavies:

                pos = atom.position
                # Indices of all gridpoints within cutoff of atom's position
                neighboridx = numpy.array(tree.query_ball_point(pos, self.cutoff))
                neighborpts = gridpts[neighboridx]

                dist_vectors = neighborpts[:, ...] - pos

                # Distance array between atom and neighbor grid points
                #distarr = scipy.spatial.distance.cdist(pos.reshape(1,3), neighborpts,
                #                                       'sqeuclidean').reshape(neighboridx.shape)

                phivals = phi(dist_vectors, self.sigma, sigma_sq, self.cutoff, cutoff_sq)

                rho_prot[i-self.start_frame, neighboridx] += phivals

            for atom in water_ow:
                pos = atom.position
                neighboridx = numpy.array(tree.query_ball_point(pos, self.cutoff))
                neighborpts = gridpts[neighboridx]

                dist_vectors = neighborpts[:, ...] - pos
                # Distance array between atom and neighbor grid points
                # distarr = scipy.spatial.distance.cdist(pos.reshape(1,3),
                #       neighborpts,'sqeuclidean').reshape(neighboridx.shape)

                phivals = phi(dist_vectors, self.sigma, sigma_sq, self.cutoff, cutoff_sq)

                rho_water[i-self.start_frame, neighboridx] += phivals

            rho[i-self.start_frame, :] = rho_prot[i-self.start_frame, :]/self.rho_prot_bulk \
                + rho_water[i-self.start_frame, :]/self.rho_water_bulk
            

        return rho

    def go(self):

        prot_heavies = self.univ.select_atoms("not (name H* or resname SOL) and not (name CL or name NA)")
        water_ow = self.univ.select_atoms("name OW")
        ions = self.univ.select_atoms("name CL or name NA")

        grid_dl = 1
        natoms = self.univ.coord.n_atoms

        # Numpy 6d array of xyz dimensions - I assume last three dimensions are axis angles?
        # should modify in future to accomodate non cubic box
        #   In angstroms
        box = self.univ.dimensions


        # Set up marching cube stuff - grids in Angstroms
        #  ngrids are grid dimensions of discretized space at resolution ngrids[i] in each dimension
        ngrids = box[:3].astype(int)+1
        dgrid = box[:3]/(ngrids-1)

        # Extra number of grid points on each side to reflect pbc
        #    In grid units (i.e. inverse dgrid)
        #margin = (cutoff/dgrid).astype(int)
        margin = numpy.array([0,0,0], dtype=numpy.int)
        # Number of actual unique points
        npts = ngrids.prod()
        # Number of real points plus margin of reflected pts
        n_pseudo_pts = (ngrids + 2*margin).prod()

        # Construct 'gridpts' array, over which we will perform
        #    nearest neighbor searching for each heavy prot and water atom
        #  NOTE: gridpts is an augmented array - that is, it includes
        #  (margin+2) array points in each dimension - this is to reflect pbc
        #  conditions - i.e. atoms near the edge of the box can 'see' grid points
        #    within a distance (opp edge +- cutoff)
        #  I use an index mapping array to (a many-to-one mapping of gridpoint to actual box point)
        #     to retrieve appropriate real point indices
        coord_x = numpy.arange(-margin[0],ngrids[0]+margin[0],dgrid[0])
        coord_y = numpy.arange(-margin[1],ngrids[1]+margin[1],dgrid[1])
        coord_z = numpy.arange(-margin[2],ngrids[2]+margin[2],dgrid[2])
        xpts, ypts, zpts = numpy.meshgrid(coord_x,coord_y,coord_z)

        # gridpts array shape: (n_pseudo_pts, 3)
        #   gridpts npseudo unique points - i.e. all points
        #      on an enlarged grid
        gridpts = numpy.array(zip(ypts.ravel(),xpts.ravel(),zpts.ravel()))
        grididx = gridpts / dgrid
        # Many-to-one mapping of pseudo pt inx => actual point idx
        #   e.g. to translate from idx of gridpts array to point in
        #      'rho' arrays, below

        # Split up frames, assign to work manager, splice back together into
        #   total rho array
        rho = self.calc_rho(npts, gridpts, prot_heavies, water_ow)

        # Hack out the last frame to a volumetric '.dx' format (readable by VMD)
        prot_tree = scipy.spatial.cKDTree(prot_heavies.positions)
        print "Average rho = {}".format(rho.mean())
        rho_shape = (rho.sum(axis=0)/rho.shape[0]).reshape(ngrids)
        #rho_shape = rho[0].reshape(ngrids)
        cntr = 0

        print "Preparing to output data"

        with open(self.output_filename, 'w') as f:
            f.write("object 1 class gridpositions counts {} {} {}\n".format(ngrids[0], ngrids[1], ngrids[2]))
            f.write("origin {:1.8e} {:1.8e} {:1.8e}\n".format(0,0,0))
            f.write("delta {:1.8e} {:1.8e} {:1.8e}\n".format(dgrid[0], 0, 0))
            f.write("delta {:1.8e} {:1.8e} {:1.8e}\n".format(0, dgrid[1], 0))
            f.write("delta {:1.8e} {:1.8e} {:1.8e}\n".format(0, 0, dgrid[2]))
            f.write("object 2 class gridconnections counts {} {} {}\n".format(ngrids[0], ngrids[1], ngrids[2]))
            f.write("object 3 class array type double rank 0 items {} data follows\n".format(npts))
            for i in xrange(ngrids[0]):
                for j in xrange(ngrids[1]):
                    for k in xrange(ngrids[2]):
                        grid_pt = numpy.array([dgrid[0]*i, dgrid[1]*j, dgrid[2]*k])
                        dist, idx = prot_tree.query(grid_pt, distance_upper_bound=10.0)

                        if dist == float('inf'):
                            rho_shape[i,j,k] += 1.0
                        #if (i < lowercutoff_x or j < lowercutoff_y or k < lowercutoff_z) or (i > uppercutoff_x or j > uppercutoff_y or k > uppercutoff_z):
                        #    rho_shape[i,j,k] += 1.0
                        f.write("{:1.8e} ".format(rho_shape[i,j,k]))
                        cntr += 1
                        if (cntr % 3 == 0):
                            f.write("\n")


if __name__=='__main__':
    Interface().main()


