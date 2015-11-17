#Instantaneous interface
# nrego sept 2015

from __future__ import division; __metaclass__ = type
import sys
import numpy
import argparse
import logging

import MDAnalysis
from MDAnalysis.coordinates.xdrfile.libxdrfile2 import read_xtc_natoms, xdrfile_open

from scipy.spatial import cKDTree
from skimage import measure

from utils import phi
from mdtools import ParallelTool

log = logging.getLogger('mdtools.interface')

def _calc_rho(lb, ub, prot_heavies, water_ow, cutoff, sigma, gridpts, npts, rho_prot_bulk, rho_water_bulk):    
    cutoff_sq = cutoff**2
    sigma_sq = sigma**2
    block = ub - lb
    rho_prot_slice = numpy.zeros((block, npts), dtype=numpy.float32)
    rho_water_slice = numpy.zeros((block, npts), dtype=numpy.float32)
    rho_slice = numpy.zeros((block, npts), dtype=numpy.float32)

    # KD tree for nearest neighbor search
    tree = cKDTree(gridpts)

    for i in xrange(block):

        for pos in prot_heavies[i]:

            #pos = atom.position
            # Indices of all gridpoints within cutoff of atom's position
            neighboridx = numpy.array(tree.query_ball_point(pos, cutoff))
            if neighboridx.size == 0:
                continue
            neighborpts = gridpts[neighboridx]

            dist_vectors = neighborpts[:, ...] - pos

            # Distance array between atom and neighbor grid points
            #distarr = scipy.spatial.distance.cdist(pos.reshape(1,3), neighborpts,
            #                                       'sqeuclidean').reshape(neighboridx.shape)

            phivals = phi(dist_vectors, sigma, sigma_sq, cutoff, cutoff_sq)

            rho_prot_slice[i, neighboridx] += phivals

        for pos in water_ow[i]:
            neighboridx = numpy.array(tree.query_ball_point(pos, cutoff))
            if neighboridx.size == 0:
                continue
            neighborpts = gridpts[neighboridx]

            dist_vectors = neighborpts[:, ...] - pos
            # Distance array between atom and neighbor grid points
            # distarr = scipy.spatial.distance.cdist(pos.reshape(1,3),
            #       neighborpts,'sqeuclidean').reshape(neighboridx.shape)

            phivals = phi(dist_vectors, sigma, sigma_sq, cutoff, cutoff_sq)

            rho_water_slice[i, neighboridx] += phivals

        # Can probably move this out of here and perform at end
        rho_slice[i, :] = rho_prot_slice[i, :]/rho_prot_bulk \
            + rho_water_slice[i, :]/rho_water_bulk

    return (rho_slice, lb, ub)

class Interface(ParallelTool):
    prog='interface'
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

        self.dgrid = None
        self.ngrids = None
        self.gridpts = None
        self.npts = None

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
            print "Error processing input files: {} and {}".format(args.grofile, args.trajfile)
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

        self.npts = 0

        self.output_filename = args.outfile

    def calc_rho(self):

        rho_water = numpy.zeros((self.n_frames, self.npts), dtype=numpy.float32)
        rho_prot = numpy.zeros((self.n_frames, self.npts), dtype=numpy.float32)
        rho = numpy.zeros((self.n_frames, self.npts), dtype=numpy.float32)

        # Cut that shit up to send to work manager
        n_workers = self.work_manager.n_workers or 1

        blocksize = self.n_frames // n_workers
        log.info('n workers: {}'.format(n_workers))
        log.info('n frames: {}'.format(self.n_frames))
        if self.n_frames % n_workers > 0:
            blocksize += 1
        log.info('blocksize: {}'.format(blocksize))

        def task_gen():
            prot_heavies = self.univ.select_atoms("not (name H* or resname SOL) and not (name CL or name NA)")
            water_ow = self.univ.select_atoms("name OW")

            if __debug__:
                checkset = set()
            for lb_frame in xrange(self.start_frame, self.last_frame, blocksize):
                ub_frame = min(self.last_frame, lb_frame+blocksize)
                lb = lb_frame - self.start_frame
                ub = ub_frame - self.start_frame
                if __debug__:
                    checkset.update(set(xrange(lb_frame,ub_frame)))

                prot_heavies_pos = numpy.zeros((ub-lb, len(prot_heavies), 3), dtype=numpy.float32)
                water_ow_pos = numpy.zeros((ub-lb, len(water_ow), 3), dtype=numpy.float32)
                for i, frame_idx in enumerate(xrange(lb_frame, ub_frame)):
                    self.univ.trajectory[frame_idx]
                    prot_heavies_pos[i, ...] = prot_heavies.positions
                    water_ow_pos[i, ...] = water_ow.positions

                args = ()
                kwargs = dict(lb=lb, ub=ub, prot_heavies=prot_heavies_pos, water_ow=water_ow_pos,
                              cutoff=self.cutoff, sigma=self.sigma, gridpts=self.gridpts, npts=self.npts, 
                              rho_prot_bulk=self.rho_prot_bulk, rho_water_bulk=self.rho_water_bulk)
                log.info("Sending job batch (from frame {} to {})".format(lb_frame, ub_frame))
                yield (_calc_rho, args, kwargs)

            if __debug__:
                assert checkset == set(xrange(self.start_frame, self.last_frame)), 'frames missing: {}'.format(set(xrange(self.start_frame, self.last_frame)) - checkset)

        # Splice together results into final array of densities
        for future in self.work_manager.submit_as_completed(task_gen(), queue_size=self.max_queue_len):
            rho_slice, lb, ub = future.get_result(discard=True)
            rho[lb:ub, :] = rho_slice
            del rho_slice
            

        return rho

    def go(self):

        grid_dl = 1
        natoms = self.univ.coord.n_atoms

        # Numpy 6d array of xyz dimensions - I assume last three dimensions are axis angles?
        # should modify in future to accomodate non cubic box
        #   In angstroms
        box = self.univ.dimensions


        # Set up marching cube stuff - grids in Angstroms
        #  ngrids are grid dimensions of discretized space at resolution ngrids[i] in each dimension
        self.ngrids = ngrids = box[:3].astype(int)+1
        self.dgrid = dgrid = box[:3]/(ngrids-1)

        log.info("Box: {}".format(box))
        log.info("Ngrids: {}".format(ngrids))

        # Extra number of grid points on each side to reflect pbc
        #    In grid units (i.e. inverse dgrid)
        #margin = (cutoff/dgrid).astype(int)
        margin = numpy.array([0,0,0], dtype=numpy.int)
        # Number of actual unique points
        self.npts = npts = ngrids.prod()
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
        self.gridpts = gridpts = numpy.array(zip(ypts.ravel(),xpts.ravel(),zpts.ravel()))
        
        gridpts = numpy.zeros((ngrids[0], ngrids[1], ngrids[2], 3))
        for i in xrange(ngrids[0]):
            for j in xrange(ngrids[1]):
                for k in xrange(ngrids[2]):
                    gridpts[i,j,k,:] = numpy.array([i,j,k]) * dgrid

        self.gridpts = gridpts = gridpts.reshape((npts, 3))
        #self.gridpts.dtype = numpy.float32
        

        log.info("Point grid set up")

        # Split up frames, assign to work manager, splice back together into
        #   total rho array
        rho = self.calc_rho()

        self.univ.trajectory[self.last_frame-1]
        prot_heavies = self.univ.select_atoms("not (name H* or resname SOL) and not (name CL or name NA)")
        # Hack out the last frame to a volumetric '.dx' format (readable by VMD)
        # Note we artificially add to all grid points more than 10 A from protein
        #   heavy atoms to remove errors at box edges - hackish, but seems to work ok
        prot_tree = cKDTree(prot_heavies.positions)
        log.info("Average rho = {}".format(rho.mean()))
        rho_shape = (rho.sum(axis=0)/rho.shape[0]).reshape(ngrids)
        #rho_shape = rho[0].reshape(ngrids)
        cntr = 0

        log.info("Preparing to output data")

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


