#Instantaneous interface
# nrego sept 2015

from __future__ import division; __metaclass__ = type
import sys
import numpy as np
from math import sqrt
import argparse
from argparse import ArgumentTypeError
import logging

from IPython import embed

import MDAnalysis
from MDAnalysis import SelectionError
#from MDAnalysis.coordinates.xdrfile.libxdrfile2 import read_xtc_natoms, xdrfile_open

from scipy.spatial import cKDTree
import itertools
#from skimage import measure
import mdtraj

from rhoutils import rho, cartesian
from mdtools import ParallelTool

from selection_specs import sel_spec_heavies, sel_spec_heavies_nowall

from fieldwriter import RhoField

log = logging.getLogger('mdtools.temporal_interface')

## Try to avoid round-off errors as much as we can...
rho_dtype = np.float32

def _calc_rho(frame_idx, excluded_indices, water_ow, dgrid, gridpts, npts, rho_water_bulk, tree, max_water_dist, min_water_dist):  

    # Length of grid voxel
    grid_cutoff = dgrid[0] / 2.0
    assert dgrid[0] == dgrid[1] == dgrid[2]  

    rho_slice = np.zeros((npts,), dtype=rho_dtype)

    water_tree = cKDTree(water_ow)
    # should only give exactly 0 or 1 gridpoint
    water_neighbors = water_tree.query_ball_tree(tree, grid_cutoff, p=float('inf'))

    for atm_idx, pos in enumerate(water_ow):
        neighboridx = np.array(water_neighbors[atm_idx])
        
        if neighboridx.size == 0:
            continue
        else:
            assert rho_slice[neighboridx].sum() == 0
            rho_slice[neighboridx] += 1

    del water_tree, water_neighbors
    
    rho_slice[excluded_indices] = rho_water_bulk

    return (rho_slice, frame_idx)

class TemporalInterface(ParallelTool):
    prog='temporal_interface'
    description = '''\
Perform temporally-averaged interface analysis on simulation data. Requires 
GROFILE and TRAJECTORY (XTC or TRR). Conduct time-averaged interface 
analysis over specified trajectory range by .

This tool supports parallelization (see options below)


-----------------------------------------------------------------------------
Command-line options
-----------------------------------------------------------------------------
'''
    
    def __init__(self):
        super(TemporalInterface,self).__init__()
        
        # Parallel processing by default (this is not actually necessary, but it is
        # informative!)
        self.wm_env.default_work_manager = self.wm_env.default_parallel_work_manager

        self.univ = None

        self.rho_water_bulk = None

        self.rho = None
        self.rho_avg = None

        self.max_water_dist = None
        self.min_water_dist = None

        self.dgrid = None
        self.grid_dl = None
        self.ngrids = None
        self.gridpts = None
        self.npts = None

        self.tree = None

        self.start_frame = None
        self.last_frame = None

        self.outdx = None
        self.outgro = None
        self.outxtc = None
        self.outpdb = None

        self.box = None

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
        sgroup.add_argument('-b', '--start', type=int, default=0,
                            help='First timepoint (in ps)')
        sgroup.add_argument('-e', '--end', type=int, 
                            help='Last timepoint (in ps)')
        sgroup.add_argument('--max-water-dist', type=float, default=13,
                            help='maximum distance (in A), from the solute, for which to select voxels to calculate density for each frame. Default 13 A. Sets normalized density to 1.0 for any voxels outside this distance')
        sgroup.add_argument('--min-water-dist', type=float, default=3,
                            help='minimum distance (in A), from solute, for which to select voxels to calculate density for. Default 3 A. Sets normalized density to 1.0 for any voxels closer than this distance to the solute')
        sgroup.add_argument('--wall', action='store_true', 
                            help='If true, consider interace near walls (default False)')
        sgroup.add_argument('--mol-sel-spec', type=str,
                            help='A custom string specifier for selecting solute atoms, if desired')
        sgroup.add_argument('--rho-water-bulk', type=float, default=0.033,
                            help='Water density to normalize by, in waters per cubic Angstrom (default is 0.033 waters/A^3)')
        agroup = parser.add_argument_group('other options')
        agroup.add_argument('-odx', '--outdx', default='interface.dx',
                        help='Output file to write instantaneous interface')
        agroup.add_argument('-ogro', '--outgro',
                        help='output file to write instantaneous interface as GRO file')
        agroup.add_argument('-oxtc', '--outxtc',
                        help='Output file to write trajectory of instantaneous interfaces as XTC file')
        agroup.add_argument('-opdb', '--outpdb', default='voxels.pdb',
                        help='Output file to write pdb file of voxels')

    def process_args(self, args):

        try:
            self.univ = u = MDAnalysis.Universe(args.grofile, args.trajfile)
        except:
            print "Error processing input files: {} and {}".format(args.grofile, args.trajfile)
            sys.exit()

        ## TODO: check this.
        self.rho_water_bulk = args.rho_water_bulk

        if (args.start > (u.trajectory.n_frames * u.trajectory.dt)):
            raise ValueError("Error: provided start time ({} ps) is greater than total time ({} ps)"
                             .format(args.start, (u.trajectory.n_frames * u.trajectory.dt)))

        self.start_frame = int(args.start * u.trajectory.dt)
        if args.end is not None:
            self.last_frame = int(args.end * u.trajectory.dt)
        else:
            self.last_frame = u.trajectory.n_frames

        self.npts = 0

        self.max_water_dist = args.max_water_dist
        self.min_water_dist = args.min_water_dist

        self.outdx = args.outdx
        self.outgro = args.outgro
        self.outxtc = args.outxtc
        self.outpdb = args.outpdb

        self.wall = args.wall
        if args.mol_sel_spec is not None:
            self.mol_sel_spec = args.mol_sel_spec
            try:
                self.univ.select_atoms(self.mol_sel_spec)
            except SelectionError:
                raise ArgumentTypeError('invalid molecule selection spec: {}'.format(args.mol_sel_spec))

        elif self.wall:
            self.mol_sel_spec = sel_spec_heavies
        else:
            self.mol_sel_spec = sel_spec_heavies_nowall

        self.init_from_args = True

        self._setup_grid()

    def calc_rho(self):

        self.rho = np.zeros((self.n_frames, self.npts), dtype=rho_dtype)

        # Cut that shit up to send to work manager
        try:
            n_workers = self.work_manager.n_workers or 1
        except AttributeError:
            n_workers = 1

        log.info('n workers: {}'.format(n_workers))
        log.info('n frames: {}'.format(self.n_frames))

        prot_heavies = self.univ.select_atoms(self.mol_sel_spec)
        prot_pos_initial = prot_heavies.positions
        prot_tree = cKDTree(prot_pos_initial)

        # find all gridpoints within max_water_dist A of solute atoms
        #   so we can set the (normalized) rho to 1.0 (to take care of edge effects due to, e.g. v-l interfaces)
        neighbor_list_by_point = prot_tree.query_ball_tree(self.tree, r=self.max_water_dist+1)
        neighbor_list = itertools.chain(*neighbor_list_by_point)
        # neighbor_idx is a unique list of all grid_pt *indices* that are within max_water_dist from
        #   *any* atom in prot_heavies
        neighbor_idx = np.unique( np.fromiter(neighbor_list, dtype=int) )

        # Find indices of all grid points that are *farther* than max_water_dist A from solute
        far_pt_idx = np.setdiff1d(np.arange(self.npts), neighbor_idx)

        # Now find all gridpoints **closer** than min_water_dist A to solute
        #   so we can set the rho to 1.0 (so we don't produce an interface over the solute's excluded volume)
        neighbor_list_by_point = prot_tree.query_ball_tree(self.tree, r=self.min_water_dist)
        neighbor_list = itertools.chain(*neighbor_list_by_point)
        # close_pt_idx is a unique list of all grid_pt *indices* that are within r=3 from
        #   *any* atom in prot_heavies
        close_pt_idx = np.unique( np.fromiter(neighbor_list, dtype=int) ) 

        self.excluded_indices = np.append(far_pt_idx, close_pt_idx)
        assert self.excluded_indices.shape[0] == far_pt_idx.shape[0] + close_pt_idx.shape[0]
        
        def task_gen():

            for frame_idx in xrange(self.start_frame, self.last_frame):

                self.univ.trajectory[frame_idx]
                water_ow = self.univ.select_atoms("name OW and around {} ({})".format(self.max_water_dist+3, self.mol_sel_spec))
                #water_ow = self.univ.select_atoms('name OW')
                water_ow_pos = water_ow.positions

                args = ()
                kwargs = dict(frame_idx=frame_idx, excluded_indices=self.excluded_indices, water_ow=water_ow_pos,
                              dgrid=self.dgrid, gridpts=self.gridpts, npts=self.npts, 
                              rho_water_bulk=self.rho_water_bulk, tree=self.tree, 
                              max_water_dist=self.max_water_dist, min_water_dist=self.min_water_dist)
                log.info("Sending job (frame {})".format(frame_idx))
                yield (_calc_rho, args, kwargs)

        # Splice together results into final array of densities
        #for future in self.work_manager.submit_as_completed(task_gen(), queue_size=self.max_queue_len):
        for future in self.work_manager.submit_as_completed(task_gen(), queue_size=n_workers):
            #import pdb; pdb.set_trace()
            rho_slice, frame_idx = future.get_result(discard=True)
            self.rho[frame_idx-self.start_frame, :] = rho_slice
            del rho_slice

        self.rho /= self.rho_water_bulk
        #for (fn, args, kwargs) in task_gen():
        #    rho_slice, frame_idx = fn(*args, **kwargs) 
        #    self.rho[frame_idx-self.start_frame, :] = rho_slice
        #    del rho_slice  

    def get_rho_avg(self):
        if self.rho is None:
            log.warning('Rho has not been calculated yet - must run calc_rho')
            return

        self.rho_avg = self.rho.mean(axis=0)

        #non_excluded_indices = self.rho_avg < 0.1
        #non_excluded_rho_avg = self.rho_avg[non_excluded_indices]

        min_rho = self.rho_avg.min()
        max_rho = self.rho_avg.max()
        mean_rho = self.rho_avg.mean()
        log.info("Min rho: {}, Max rho: {}, avg rho: {}".format(min_rho, max_rho, mean_rho))



    def _setup_grid(self):
        grid_dl = 1 # Grid resolution at *most* 1 angstrom (when box dimension an integral number of angstroms)
        natoms = self.univ.coord.n_atoms

        # np 6d array of xyz dimensions - I assume last three dimensions are axis angles?
        # should modify in future to accomodate non cubic box
        #   In angstroms
        self.box = box = np.ceil(self.univ.dimensions[:3])

        # Set up marching cube stuff - grids in Angstroms
        #  ngrids are grid dimensions of discretized space at resolution ngrids[i] in each dimension
        self.ngrids = ngrids = (box/ grid_dl).astype(int)+1
        self.dgrid = dgrid = np.ones(3) * grid_dl

        log.info("Box: {}".format(box))
        log.info("Ngrids: {}".format(ngrids))

        # Extra number of grid points on each side to reflect pbc
        #    In grid units (i.e. inverse dgrid)
        #margin = (cutoff/dgrid).astype(int)
        margin = np.array([0,0,0], dtype=np.int)
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
        coord_x = np.linspace(-margin[0],box[0]+margin[0],ngrids[0])
        coord_y = np.linspace(-margin[1],box[1]+margin[1],ngrids[1])
        coord_z = np.linspace(-margin[2],box[2]+margin[2],ngrids[2])

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

        writer = RhoField(self.rho_shape, self.gridpts)

        # Always do dx file output
        writer.do_DX(self.outdx)

        if self.outgro is not None:
            writer.do_GRO(self.outgro, frame=0)

        if self.outxtc is not None:
            writer.do_XTC(self.outxtc)

    #TODO: move this elsewhere
    def do_pdb_output(self):
        self.univ.trajectory[0]
        prot_atoms = self.univ.select_atoms(self.mol_sel_spec)
        prot_tree = cKDTree(prot_atoms.positions)

        neighbor_list_by_point = prot_tree.query_ball_tree(self.tree, r=self.max_water_dist)
        neighbor_list = itertools.chain(*neighbor_list_by_point)
        # neighbor_idx is a unique list of all grid_pt *indices* that are within max_water_dist from
        #   *any* atom in prot_heavies
        neighbor_idx = np.unique( np.fromiter(neighbor_list, dtype=int) )


        bfactors = 100*(1 - np.clip(self.rho_avg[neighbor_idx], 0.0, 1.0))
        bfactors = np.clip(bfactors, 0, 99.99)
        
        top = mdtraj.Topology()
        c = top.add_chain()

        cnt = 0
        for i in range(neighbor_idx.size):
            cnt += 1
            r = top.add_residue('II', c)
            a = top.add_atom('II', mdtraj.element.get_by_symbol('VS'), r, i)

        with mdtraj.formats.PDBTrajectoryFile(self.outpdb, 'w') as f:
            # Mesh pts have to be in nm
            f.write(self.gridpts[neighbor_idx], top, bfactors=bfactors)


    def go(self):

        #self.setup_grid()
        # Split up frames, assign to work manager, splice back together into
        #   total rho array
        self.calc_rho()
        self.get_rho_avg()
        #embed()
        self.do_output()
        self.do_pdb_output()


if __name__=='__main__':
    TemporalInterface().main()


