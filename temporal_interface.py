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
from mdtools import ParallelTool, Subcommand

from selection_specs import sel_spec_heavies, sel_spec_heavies_nowall

from fieldwriter import RhoField

log = logging.getLogger('mdtools.temporal_interface')

## Try to avoid round-off errors as much as we can...
rho_dtype = np.float32

def _calc_rho(frame_idx, water_ow, dgrid, gridpts, npts, rho_water_bulk, tree):  

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

        elif neighboridx.size >= 2:

            rho_slice[neighboridx[0]] += 1
        elif neighboridx.size == 1:
            rho_slice[neighboridx] += 1


    del water_tree, water_neighbors

    return (rho_slice, frame_idx)


class TemporalInterfaceSubcommand(Subcommand):
    ''' Note this will inherit a work manager from parent (which is always the TemporalInterface main instance.
    Common args and methods for any Temporal Interface/Density subcommands; i.e. trajectory input processing, 
    Rho calculation and averaging (assuming the inheriting subclass initializes the voxel grid appropriately'''

    # Expect that derived subcommand classes will fill these in.
    #   Specifies how to fill out subparser
    subcommand = None
    help_text = None
    description = None

    def __init__(self, parent):
        super(TemporalInterfaceSubcommand,self).__init__(parent)

        self.univ = None

        self.rho_water_bulk = None

        self.rho = None
        self.rho_avg = None

        self.max_water_dist = None
        self.min_water_dist = None

        self.x_bounds = None
        self.y_bounds = None
        self.z_bounds = None
        self.gridpts = None

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
        
        sgroup = parser.add_argument_group('Trajectory Input Options')
        sgroup.add_argument('-c', '--grofile', metavar='INPUT', type=str, required=True,
                            help='Input structure file')
        sgroup.add_argument('-f', '--trajfile', metavar='XTC', type=str, required=True,
                            help='Input XTC trajectory file')
        sgroup.add_argument('-b', '--start', type=int, default=0,
                            help='First timepoint (in ps)')
        sgroup.add_argument('-e', '--end', type=int, 
                            help='Last timepoint (in ps)')
        sgroup.add_argument('--mol-sel-spec', type=str,
                            help='A custom string specifier for selecting solute atoms, if desired')


    def process_args(self, args):

        try:
            self.univ = u = MDAnalysis.Universe(args.grofile, args.trajfile)
        except:
            print "Error processing input files: {} and {}".format(args.grofile, args.trajfile)
            sys.exit()

        assert np.array_equal(self.univ.dimensions[3:], np.array([90.,90.,90.])), "not a cubic box!"

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

        self.outdx = args.outdx
        self.outgro = args.outgro
        self.outxtc = args.outxtc
        self.outpdb = args.outpdb

        if args.mol_sel_spec is not None:
            self.mol_sel_spec = args.mol_sel_spec
            try:
                self.univ.select_atoms(self.mol_sel_spec)
            except SelectionError:
                raise ArgumentTypeError('invalid molecule selection spec: {}'.format(args.mol_sel_spec))

    def calc_rho(self):

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
                water_ow = self.univ.select_atoms("name OW and around {} ({})".format(self.max_water_dist+3, self.mol_sel_spec))
                #water_ow = self.univ.select_atoms('name OW')
                water_ow_pos = water_ow.positions

                args = ()
                kwargs = dict(frame_idx=frame_idx, water_ow=water_ow_pos,
                              dgrid=self.dgrid, gridpts=self.gridpts, npts=self.npts, 
                              rho_water_bulk=self.rho_water_bulk, tree=self.tree)
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

    def get_rho_avg(self):
        if self.rho is None:
            log.warning('Rho has not been calculated yet - must run calc_rho')
            return

        self.rho_avg = self.rho.mean(axis=0)
        log.info("total : {}".format(self.rho_avg.sum()))

        #non_excluded_indices = self.rho_avg < 0.1
        #non_excluded_rho_avg = self.rho_avg[non_excluded_indices]

        min_rho = self.rho_avg.min()
        max_rho = self.rho_avg.max()
        mean_rho = self.rho_avg.mean()
        log.info("Min rho: {}, Max rho: {}, avg rho: {}, avg rho*npts: {}".format(min_rho, max_rho, mean_rho, mean_rho*self.npts))


    @property
    def npts(self):
        return self.gridpts.shape[0]

    #TODO: move this elsewhere
    def do_pdb_output(self):

        norm_rho_p = 1 - self.rho_avg
        np.savetxt('norm_rho_p.dat', norm_rho_p)
        np.savetxt('avg_rho', self.rho_avg*self.rho_water_bulk)
        n_depleted = np.sum(norm_rho_p) * self.rho_water_bulk
        n_avg = np.sum(self.rho_avg) * self.rho_water_bulk
        header = "<n>_phi   (<n>_0 - <n>_phi) npts"
        myarr = np.array([[n_avg, n_depleted, self.npts]])
        np.savetxt("navg_ndep.dat", myarr, header=header)
        print("number depleted: {}".format(n_depleted))
        bfactors = 100*(1 - self.rho_avg)
        bfactors = np.clip(bfactors, 0, 100)
        
        top = mdtraj.Topology()
        c = top.add_chain()

        cnt = 0
        for i in range(self.npts):
            cnt += 1
            r = top.add_residue('II', c)
            a = top.add_atom('II', mdtraj.element.get_by_symbol('VS'), r, i)

        with mdtraj.formats.PDBTrajectoryFile(self.outpdb, 'w') as f:
            # Mesh pts have to be in nm
            f.write(self.gridpts, top, bfactors=bfactors)

    def setup_grid(self):
        ''' Derived classes must figure out how to appropriately initialize voxel grid'''
        raise NotImplementedError

    def go(self):

        #self.setup_grid()
        # Split up frames, assign to work manager, splice back together into
        #   total rho array
        self.setup_grid()
        self.calc_rho()
        self.get_rho_avg()
        #embed()
        #self.do_output()
        self.do_pdb_output()


class TemporalInterfaceInitSubcommand(TemporalInterfaceSubcommand):
    subcommand='init'
    help_text='Initialize voxel grid, determine voxels to include/exclude, and normalized rhos for each voxel'
    description = '''\
Initialize voxel grid from input args, calculte rho for each voxel from input data, and output voxel limits in each dimension, 
a list of excluded voxels, and the rho for each included voxel. All output to be used in subsequent analysis using the 'anal'
command
            input args:
                a) a structure and trajectory file for (presumably) an unbiased simulation
                b) a solute definition (by --molspec)
                c) a maximum distance (from solute); any voxels more than this distance are excluded
                d) a minimum distance (from solute); any voxels less than this distance are excluded

              Note: after excluding all voxels acccording to supplied min and max distances, any voxels that have a rho_i==0 will
                 also be excluded from the final voxel list
            outputs:
                a) list of x, y and z limits of voxels (determined by initial box size)
                b) boolean array of excluded, included indices (this, along with limits, above, can be used to reconstruct all included voxels)
                c) list of rho_i's (from above) for each included voxel

    '''

    def __init__(self, parent):
        super(TemporalInterfaceInitSubcommand,self).__init__(parent)

        self.grid_resolution = None
        # mask of gridpoints to include
        self.grid_mask = None

    def add_args(parser):
        group = parser.add_argument_group('Grid initialization options')
        group.add_argument('--max-water-dist', type=float, default=6,
                            help='maximum distance (in A), from the solute, for which to select voxels to calculate density for each frame. Default 13 A. Sets normalized density to 1.0 for any voxels outside this distance')
        group.add_argument('--min-water-dist', type=float, default=0,
                            help='minimum distance (in A), from solute, for which to select voxels to calculate density for. Default 0 A. Sets normalized density to 1.0 for any voxels closer than this distance to the solute')
        group.add_argument('--grid-resolution', type=float, default=1.0,
                            help='Grid resolution (in A). Will construct initial grid in order to completely fill first frame box with an integral number of voxels')

    def process_args(args):
        self.grid_resolution = args.grid_resolution
        self.min_water_dist = args.min_water_dist
        self.max_water_dist = args.max_water_dist

    def setup_grid(self):
        
        n_atoms = self.univ.coord.n_atoms
        self.box = box = np.ceil(self.univ.dimensions[:3])

        # Set up marching cube stuff - grids in Angstroms
        #  ngrids are grid dimensions of discretized space at resolution ngrids[i] in each dimension
        self.ngrids = ngrids = (box / self.grid_resolution).astype(int)+1

        log.info("Initializing grid from initial frame")
        log.info("Box: {}".format(box))
        log.info("Ngrids: {}".format(ngrids))

        # Construct 'gridpts' array, over which we will perform
        #    nearest neighbor searching for each heavy prot and water atom
        #  NOTE: gridpts is an augmented array - that is, it includes
        #  (margin+2) array points in each dimension - this is to reflect pbc
        #  conditions - i.e. atoms near the edge of the box can 'see' grid points
        #    within a distance (opp edge +- cutoff)
        #  I use an index mapping array to (a many-to-one mapping of gridpoint to actual box point)
        #     to retrieve appropriate real point indices
        self.x_bounds = np.linspace(0,box[0],n_grids[0])
        self.y_bounds = np.linspace(0,box[1],n_grids[1])
        self.z_bounds = np.linspace(0,box[2],n_grids[2])

        # gridpts array shape: (n_pts, 3)
        #   gridpts npseudo unique points - i.e. all points
        #      on an enlarged grid
        self.gridpts = gridpts = cartesian([self.x_bounds, self.y_bounds, self.z_bounds])
        tree = cKDTree(self.gridpts)

        # Exclude all gridpoints less than min_dist to protein and morre than max dist from prot
        prot_heavies = self.univ.select_atoms(self.mol_sel_spec)
        prot_pos_initial = prot_heavies.positions
        prot_tree = cKDTree(prot_pos_initial)

        # find all gridpoints within max_water_dist A of solute atoms
        #   so we can set the (normalized) rho to 1.0 (to take care of edge effects due to, e.g. v-l interfaces)
        neighbor_list_by_point = prot_tree.query_ball_tree(self.tree, r=self.max_water_dist)
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

        excluded_indices = np.append(far_pt_idx, close_pt_idx)
        assert excluded_indices.shape[0] == far_pt_idx.shape[0] + close_pt_idx.shape[0]
        
        good_indices = np.setdiff1d(np.arange(self.npts), excluded_indices)
        self.gridpts = self.gridpts[good_indices]

        self.tree = cKDTree(self.gridpts)

        log.info("Point grid set up")   

class TemporalInterfaceAnalSubcommand(TemporalInterfaceSubcommand):
    subcommand='anal'
    help_text=''
    description = '''\
Run time-averaged interace analysis on trajectory. Load pre-initialized voxel grid definitions and reference rho values for each voxel,
And then calculates average rho and normalized rho values for input data trajectory

    '''
    def __init__(self, parent):
        super(TemporalInterfaceAnalSubcommand,self).__init__(parent)


class TemporalInterface(ParallelTool):
    subcommands=[TemporalInterfaceInitSubcommand, TemporalInterfaceAnalSubcommand]
    subparsers_title= 'Temporal density-averaging modes'
    description = '''\
Conduct time-averaged density analysis on a given dataset (contrast with interface.py, which coarse-grains the density field for each 
iteration in the trajectory)
'''

    def __init__(self):
        super(TemporalInterface,self).__init__()
        self._subcommand = None
        self._avail_subcommand = {subcommand_class.subcommand: subcommand_class(self) for subcommand_class in self.subcommands}

    def add_args(self, parser):
        subparsers = parser.add_subparsers(title=self.subparsers_title)

        for instance in self._avail_subcommand.itervalues():
            instance.add_subparser(subparsers)

    def process_args(self, args):
        self._subcommand = args.subcommand
        self._subcommand.process_all_args(args)


if __name__=='__main__':
    TemporalInterface().main()


