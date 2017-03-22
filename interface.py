#Instantaneous interface
# nrego sept 2015

from __future__ import division; __metaclass__ = type
import sys
import numpy as np
from math import sqrt
import argparse
import logging

import MDAnalysis
#from MDAnalysis.coordinates.xdrfile.libxdrfile2 import read_xtc_natoms, xdrfile_open

from scipy.spatial import cKDTree
import itertools
#from skimage import measure

from rhoutils import rho, cartesian
from mdtools import ParallelTool

from selection_specs import sel_spec_heavies, sel_spec_heavies_nowall

from fieldwriter import RhoField

log = logging.getLogger('mdtools.interface')

## Try to avoid round-off errors as much as we can...
rho_dtype = np.float32

def _calc_rho(frame_idx, prot_heavies, water_ow, cutoff, sigma, gridpts, npts, rho_prot_bulk, rho_water_bulk, tree, water_cutoff):    
    cutoff_sq = cutoff**2
    sigma_sq = sigma**2
    rho_prot_slice = np.zeros((npts,), dtype=rho_dtype)
    rho_water_slice = np.zeros((npts,), dtype=rho_dtype)
    rho_slice = np.zeros((npts,), dtype=rho_dtype)

    prot_tree = cKDTree(prot_heavies)
    prot_neighbors = prot_tree.query_ball_tree(tree, cutoff, p=float('inf'))

    # position of each atom at frame i
    for atm_idx, pos in enumerate(prot_heavies):

        #pos = atom.position
        # Indices of all gridpoints within cutoff of atom's position
        neighboridx = np.array(prot_neighbors[atm_idx])
        if neighboridx.size == 0:
            continue
        neighborpts = gridpts[neighboridx]

        dist_vectors = neighborpts[:, ...] - pos
        dist_vectors = dist_vectors.astype(rho_dtype)
        # Distance array between atom and neighbor grid points
        #distarr = scipy.spatial.distance.cdist(pos.reshape(1,3), neighborpts,
        #                                       'sqeuclidean').reshape(neighboridx.shape)

        rhovals = rho(dist_vectors, sigma, sigma_sq, cutoff, cutoff_sq)
        rho_prot_slice[neighboridx] += rhovals

        del dist_vectors, neighborpts

    # find all gridpoints within 12 A of protein atoms
    #   so we can set the rho to 1.0 (to take care of edge effects due to, e.g. v-l interfaces)
    neighbor_list_by_point = prot_tree.query_ball_tree(tree, r=water_cutoff-3)
    neighbor_list = itertools.chain(*neighbor_list_by_point)
    # neighbor_idx is a unique list of all grid_pt *indices* that are within r=12 from
    #   *any* atom in prot_heavies
    neighbor_idx = np.unique( np.fromiter(neighbor_list, dtype=int) )

    # Find indices of all grid points that are *farther* than 12 A from protein
    far_pt_idx = np.setdiff1d(np.arange(npts), neighbor_idx)

    assert neighbor_idx.shape[0] + far_pt_idx.shape[0] == npts

    del prot_tree, prot_neighbors, neighbor_list_by_point, neighbor_list, neighbor_idx

    water_tree = cKDTree(water_ow)
    water_neighbors = water_tree.query_ball_tree(tree, cutoff, p=float('inf'))

    for atm_idx, pos in enumerate(water_ow):
        neighboridx = np.array(water_neighbors[atm_idx])
        if neighboridx.size == 0:
            continue
        neighborpts = gridpts[neighboridx]

        dist_vectors = neighborpts[:, ...] - pos
        dist_vectors = dist_vectors.astype(rho_dtype)

        rhovals = rho(dist_vectors, sigma, sigma_sq, cutoff, cutoff_sq)
        rho_water_slice[neighboridx] += rhovals

        del dist_vectors, neighborpts

    del water_tree, water_neighbors
    # Can probably move this out of here and perform at end
    rho_slice = rho_prot_slice/rho_prot_bulk \
        + rho_water_slice/rho_water_bulk
    rho_slice[far_pt_idx] = 1.0

    return (rho_slice, frame_idx)

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

        self.rho = None
        self.rho_avg = None

        self.dgrid = None
        self.ngrids = None
        self.gridpts = None
        self.npts = None

        self.tree = None

        self.start_frame = None
        self.last_frame = None

        self.outdx = None
        self.outgro = None
        self.outxtc = None

        # consider interfaces near walls
        self.wall = None
        # sel_spec (string) for selecting solute atoms
        self.mol_sel_spec = None

        self.init_from_args = False

        self._rho_shape = None

    @property
    def n_frames(self):
        return self.last_frame - self.start_frame

    @property
    def rho_shape(self):
        if self.rho is not None and self._rho_shape is None:
            self._rho_shape = self.rho.reshape((self.n_frames, self.ngrids[0], self.ngrids[1], self.ngrids[2]))
       
        return self._rho_shape
        
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
        sgroup.add_argument('-e', '--end', type=int, 
                            help='Last timepoint (in ps)')
        sgroup.add_argument('--wall', action='store_true', 
                            help='If true, consider interace near walls (default False)')
        sgroup.add_argument('--rhoprot', default=40, type=float,
                        help='Estimated protein density (heavy atoms per nm3)')

        agroup = parser.add_argument_group('other options')
        agroup.add_argument('-odx', '--outdx', default='interface.dx',
                        help='Output file to write instantaneous interface')
        agroup.add_argument('-ogro', '--outgro',
                        help='output file to write instantaneous interface as GRO file')
        agroup.add_argument('-oxtc', '--outxtc',
                        help='Output file to write trajectory of instantaneous interfaces as XTC file')


    def process_args(self, args):

        try:
            self.univ = u = MDAnalysis.Universe(args.grofile, args.trajfile)
        except:
            print "Error processing input files: {} and {}".format(args.grofile, args.trajfile)
            sys.exit()

        self.cutoff = args.cutoff
        self.sigma = 2.4
        self.rho_water_bulk = 0.033
        self.rho_prot_bulk = args.rhoprot / 1000.0

        if (args.start > (u.trajectory.n_frames * u.trajectory.dt)):
            raise ValueError("Error: provided start time ({} ps) is greater than total time ({} ps)"
                             .format(args.start, (u.trajectory.n_frames * u.trajectory.dt)))

        self.start_frame = int(args.start / u.trajectory.dt)
        if args.end is not None:
            self.last_frame = int(args.end / u.trajectory.dt)
        else:
            self.last_frame = u.trajectory.n_frames

        self.npts = 0

        self.outdx = args.outdx
        self.outgro = args.outgro
        self.outxtc = args.outxtc

        self.wall = args.wall
        if self.wall:
            self.mol_sel_spec = sel_spec_heavies
        else:
            self.mol_sel_spec = sel_spec_heavies_nowall

        self.init_from_args = True

        self._setup_grid()

    #@profile
    def calc_rho(self):

        water_dist_cutoff = int(np.sqrt(3*self.cutoff**2))
        water_dist_cutoff += 1

        rho_water = np.zeros((self.n_frames, self.npts), dtype=rho_dtype)
        rho_prot = np.zeros((self.n_frames, self.npts), dtype=rho_dtype)
        self.rho = np.zeros((self.n_frames, self.npts), dtype=rho_dtype)

        # Cut that shit up to send to work manager
        try:
            n_workers = self.work_manager.n_workers or 1
        except AttributeError:
            n_workers = 1

        log.info('n workers: {}'.format(n_workers))
        log.info('n frames: {}'.format(self.n_frames))


        def task_gen():

            for frame_idx in xrange(self.start_frame, self.last_frame):

                self.univ.trajectory[frame_idx]
                prot_heavies = self.univ.select_atoms(self.mol_sel_spec)
                water_ow = self.univ.select_atoms("name OW and around {} ({})".format(water_dist_cutoff, self.mol_sel_spec))
                #water_ow = self.univ.select_atoms('name OW')
                prot_heavies_pos = prot_heavies.positions
                water_ow_pos = water_ow.positions

                args = ()
                kwargs = dict(frame_idx=frame_idx, prot_heavies=prot_heavies_pos, water_ow=water_ow_pos,
                              cutoff=self.cutoff, sigma=self.sigma, gridpts=self.gridpts, npts=self.npts, 
                              rho_prot_bulk=self.rho_prot_bulk, rho_water_bulk=self.rho_water_bulk,
                              tree=self.tree, water_cutoff=water_dist_cutoff)
                log.info("Sending job (frame {})".format(frame_idx))
                yield (_calc_rho, args, kwargs)

        # Splice together results into final array of densities
        #for future in self.work_manager.submit_as_completed(task_gen(), queue_size=self.max_queue_len):
        for future in self.work_manager.submit_as_completed(task_gen(), queue_size=n_workers):
            #import pdb; pdb.set_trace()
            rho_slice, frame_idx = future.get_result(discard=True)
            self.rho[frame_idx-self.start_frame, :] = rho_slice
            del rho_slice
        #for (fn, args, kwargs) in task_gen():
        #    rho_slice, frame_idx = fn(*args, **kwargs) 
        #    self.rho[frame_idx-self.start_frame, :] = rho_slice
        #    del rho_slice  

    def get_rho_avg(self, weights=None):
        if self.rho is None:
            log.warning('Rho has not been calculated yet - must run calc_rho')
            return
        if weights is None:
            weights = np.ones((self.n_frames))

        assert weights.size == self.rho.shape[0]

        weights /= weights.sum()

        self.rho_avg = np.sum((self.rho * weights[:, np.newaxis]), axis=0)
        min_rho = self.rho_avg.min()
        max_rho = self.rho_avg.max()
        log.info("Min rho: {}, Max rho: {}".format(min_rho, max_rho))



    def _setup_grid(self):
        grid_dl = 1 # Grid resolution at *most* 1 angstrom (when box dimension an integral number of angstroms)
        natoms = self.univ.coord.n_atoms

        # np 6d array of xyz dimensions - I assume last three dimensions are axis angles?
        # should modify in future to accomodate non cubic box
        #   In angstroms
        box = self.univ.dimensions[:3]

        # Set up marching cube stuff - grids in Angstroms
        #  ngrids are grid dimensions of discretized space at resolution ngrids[i] in each dimension
        self.ngrids = ngrids = np.ceil(box/ grid_dl).astype(int)+1
        self.dgrid = np.ones(3) * grid_dl

        log.info("Box: {}".format(box))
        log.info("Ngrids: {}".format(ngrids))

        # Number of actual unique points
        self.npts = npts = ngrids.prod()

        # Construct 'gridpts' array, over which we will perform
        #    nearest neighbor searching for each heavy prot and water atom
        #  NOTE: gridpts is an augmented array - that is, it includes
        #  (margin+2) array points in each dimension - this is to reflect pbc
        #  conditions - i.e. atoms near the edge of the box can 'see' grid points
        #    within a distance (opp edge +- cutoff)
        #  I use an index mapping array to (a many-to-one mapping of gridpoint to actual box point)
        #     to retrieve appropriate real point indices
        coord_x = np.arange(0,ngrids[0],grid_dl)
        coord_y = np.arange(0,ngrids[1],grid_dl)
        coord_z = np.arange(0,ngrids[2],grid_dl)

        # gridpts array shape: (n_pseudo_pts, 3)
        #   gridpts npseudo unique points - i.e. all points
        #      on an enlarged grid
        self.gridpts = gridpts = cartesian([coord_x, coord_y, coord_z])
        
        self.tree = cKDTree(self.gridpts)

        log.info("Point grid set up")   


    # rho: array, shape (n_frames, npts) - calculated coarse-grained rho for each grid point, for 
    #         each frame
    # weights: (optional) array, shape (n_frames,) - array of weights for averaging rho - default 
    #         is array of equal weights
    def do_output(self):

        if self.rho is None:
            log.warning("Rho not calculated - run calc_rho first")

            return

        log.info("Preparing to output data")
        #embed()
        writer = RhoField(self.rho_shape, self.gridpts)

        # Always do dx file output
        writer.do_DX(self.outdx)

        if self.outgro is not None:
            writer.do_GRO(self.outgro, frame=0)

        if self.outxtc is not None:
            writer.do_XTC(self.outxtc)


    def go(self):

        #self.setup_grid()
        # Split up frames, assign to work manager, splice back together into
        #   total rho array
        self.calc_rho()
        #embed()
        self.do_output()


if __name__=='__main__':
    Interface().main()


