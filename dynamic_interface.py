'''
Calculates time-averaged density profiles for a dynamic union of spherical probe volumes

(E.g. spheres pegged to solute heavy atoms)

"interface" may be a misnomer, and there is a lot of code copied from the other interface
   tools. 

   TODO: Refactor all interface tools to reduce code reuse; need base/core class/interface 
   to handle shared functionality (e.g. calc_rho API, data i/o, etc)
'''


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

from mdtools import ParallelTool

from constants import SEL_SPEC_HEAVIES, SEL_SPEC_HEAVIES_NOWALL

from fieldwriter import RhoField

log = logging.getLogger('mdtools.interface')
from IPython import embed

'''
Returns:
    solute_occ:  shape (n_solute_atoms,), list of ints of number of waters for each solute atom's spheres
    total_n: (int); total number of waters in the union of subvolumes
'''
def _calc_rho(frame_idx, solute_pos, water_pos, r_cutoff, neighbor_pos):    

    solute_occ = np.zeros((solute_pos.shape[0]), dtype=int)
    solute_tree = cKDTree(solute_pos)
    water_tree = cKDTree(water_pos)

    water_neighbors = solute_tree.query_ball_tree(water_tree, r=r_cutoff)
    solute_neighbors = solute_tree.query_ball_tree(solute_tree, r=r_cutoff)

    # number of water or solute atoms in each atom's subvol
    water_occ = np.array([len(neighbors) for neighbors in water_neighbors], dtype=int)
    solute_occ = np.array([len(neighbors) for neighbors in solute_neighbors], dtype=int)

    water_unique_neighbors = itertools.chain(*solute_neighbors)
    water_unique_neighbors = np.unique( np.fromiter(water_unique_neighbors, dtype=int) )

    # Total number of waters in entire probe volume
    total_n = water_unique_neighbors.size

    # Optionally find distance between each solute atom and its nearest neighbor in neighbor_pos
    neighbor_tree = cKDTree(neighbor_pos)
    min_dist = neighbor_tree.query(solute_pos)[0]


    return (water_occ, solute_occ, total_n, min_dist, frame_idx)

class DynamicInterface(ParallelTool):
    prog='dynamic interface'
    description = '''\
Perform interface/localized density analysis on a dynamic probe volume 
(i.e. a union of spherical shells, each with the same radius, centered
on user-specified atoms)

Analyses include:

- find number of waters within specified distance from each target atom 
- find the minimum neighbor distance between each target atom and another set of atoms (optional)

-----------------------------------------------------------------------------
Command-line options
-----------------------------------------------------------------------------
'''
    
    def __init__(self):
        super(DynamicInterface,self).__init__()
        
        # Parallel processing by default (this is not actually necessary, but it is
        # informative!)
        self.wm_env.default_work_manager = self.wm_env.default_parallel_work_manager

        self.univ = None

        self.r_cutoff = None
        self.rho_water_bulk = None
        self.rho_solute_bulk = None

        self._solute_atoms = None

        # Shape: (n_frames, n_solute_atoms)
        self.rho_water = None
        self.rho_solute = None
        self._avg_rho = None

        # Shape: (n_frames, n_solute_atoms)
        self.min_dist = None

        # Record of the total number of waters in the entire probe volume
        # Shape: (n_frames,)
        self.total_n = None

        self.start_frame = None
        self.last_frame = None

        self.outpdb = None
        self.outxtc = None

        self.mol_sel_spec = None

    @property
    def n_frames(self):
        return self.last_frame - self.start_frame

    @property
    def rho(self):
        return self.rho_water + self.rho_solute

    @property
    def rho_shape(self):
        if self.rho is not None and self._rho_shape is None:
            self._rho_shape = self.rho.reshape((self.n_frames, self.ngrids[0], self.ngrids[1], self.ngrids[2]))
       
        return self._rho_shape

    @property
    def solute_atoms(self):
        if self._solute_atoms is None:
            self._solute_atoms = self.univ.select_atoms(self.mol_sel_spec)

        return self._solute_atoms

    @property
    def avg_rho(self):
        if self._avg_rho is None:
            self._avg_rho = self._get_avg_rho()
        
        return self._avg_rho
    
    @property
    def n_solute_atoms(self):
        return self.solute_atoms.n_atoms
        
    def add_args(self, parser):
        
        sgroup = parser.add_argument_group('Dynamic water density options')
        sgroup.add_argument('-c', '--grofile', metavar='INPUT', type=str, required=True,
                            help='Input structure file')
        sgroup.add_argument('-f', '--trajfile', metavar='XTC', type=str, required=True,
                            help='Input XTC trajectory file')
        sgroup.add_argument('--mol-sel-spec', type=str, default=SEL_SPEC_HEAVIES,
                            help='MDAnalysis-style selection string for choosing solute atoms defining the spherical probe volume')
        sgroup.add_argument('--neighbor-sel-spec', type=str, 
                            help='MDAnalysis-style selection spec for neighbor group (such as binding partner)')
        sgroup.add_argument('-rcut', '--rcutoff', type=float, default=6.0,
                            help='Radius of individual spherical subvolumes, in A (default: 6.0)')
        sgroup.add_argument('-b', '--start', type=int, default=0,
                            help='First timepoint (in ps)')
        sgroup.add_argument('-e', '--end', type=int, 
                            help='Last timepoint (in ps)')
        sgroup.add_argument('--rhowater', type=float, default=0.033,
                            help='Mean water density to normalize occupancies, per A^3 (default: 0.033)')
        sgroup.add_argument('--rhosolute', type=float, default=0.050,
                            help='Mean solute density to normalize occupances, per A^3 (default: 0.050)')
        agroup = parser.add_argument_group('other options')
        agroup.add_argument('-opdb', '--outpdb', default='dynamic_volume.pdb',
                        help='output file to write instantaneous interface as GRO file')
        agroup.add_argument('-oxtc', '--outxtc',
                        help='Output file to write trajectory of instantaneous interfaces as XTC file')


    def process_args(self, args):

        try:
            self.univ = u = MDAnalysis.Universe(args.grofile, args.trajfile)
            self.univ.add_TopologyAttr('tempfactor')
        except:
            print "Error processing input files: {} and {}".format(args.grofile, args.trajfile)
            sys.exit()

        self.r_cutoff = args.rcutoff
        self.rho_water_bulk = args.rhowater
        self.rho_solute_bulk = args.rhosolute

        if (args.start > (u.trajectory.n_frames * u.trajectory.dt)):
            raise ValueError("Error: provided start time ({} ps) is greater than total time ({} ps)"
                             .format(args.start, (u.trajectory.n_frames * u.trajectory.dt)))

        self.start_frame = int(args.start / u.trajectory.dt)
        if args.end is not None:
            self.last_frame = int(args.end / u.trajectory.dt)
        else:
            self.last_frame = u.trajectory.n_frames


        self.outpdb = args.outpdb.split('.')[0]

        self.mol_sel_spec = args.mol_sel_spec
        self.neighbor_sel_spec = args.neighbor_sel_spec

    #@profile
    def calc_rho(self):

        water_dist_cutoff = self.r_cutoff + 2 

        self.rho_water = np.zeros((self.n_frames, self.n_solute_atoms), dtype=np.int)
        self.rho_solute = np.zeros((self.n_frames, self.n_solute_atoms), dtype=np.int)
        self.min_dist = np.zeros((self.n_frames, self.n_solute_atoms), dtype=np.float32)
        self.n_waters = np.zeros((self.n_frames, ), dtype=np.int)

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
                solute_atoms = self.univ.select_atoms(self.mol_sel_spec)
                water_atoms = self.univ.select_atoms("name OW and around {} ({})".format(water_dist_cutoff, self.mol_sel_spec))
                solute_pos = solute_atoms.positions
                water_pos = water_atoms.positions
                if self.neighbor_sel_spec is not None:
                    neighbor_atoms = self.univ.select_atoms(self.neighbor_sel_spec)
                    neighbor_pos = neighbor_atoms.positions
                else:
                    neighbor_pos = np.array([[np.inf, np.inf, np.inf]])

                args = ()
                kwargs = dict(frame_idx=frame_idx, solute_pos=solute_pos, water_pos=water_pos,
                              r_cutoff=self.r_cutoff, neighbor_pos=neighbor_pos)
                log.info("Sending job (frame {})".format(frame_idx))
                yield (_calc_rho, args, kwargs)

        # Splice together results into final array of densities
        #for future in self.work_manager.submit_as_completed(task_gen(), queue_size=self.max_queue_len):
        for future in self.work_manager.submit_as_completed(task_gen(), queue_size=n_workers):
            #import pdb; pdb.set_trace()
            water_occ_slice, solute_occ_slice, total_n, min_dist, frame_idx = future.get_result(discard=True)
            self.rho_water[frame_idx-self.start_frame, :] = water_occ_slice
            self.rho_solute[frame_idx-self.start_frame, :] = solute_occ_slice
            self.min_dist[frame_idx-self.start_frame, :] = min_dist

            self.n_waters[frame_idx-self.start_frame] = total_n
            del water_occ_slice, solute_occ_slice, min_dist


    def _get_avg_rho(self):
        if self.rho is None:
            log.warning('Rho has not been calculated yet - must run calc_rho')
            return
        #embed()
        avg_rho_water = self.rho_water.mean(axis=0)
        var_rho_water = self.rho_water.var(axis=0)
        avg_rho_solute = self.rho_solute.mean(axis=0)
        var_rho_solute = self.rho_solute.var(axis=0)
        water_norm = (4/3) * np.pi * (self.r_cutoff**3) * self.rho_water_bulk
        solute_norm = (4/3) * np.pi * (self.r_cutoff**3) * self.rho_solute_bulk
        avg_rho = (avg_rho_water / water_norm) + (avg_rho_solute / solute_norm)
        min_rho = avg_rho.min()
        max_rho = avg_rho.max()
        mean_rho = avg_rho.mean()
        log.info("Min rho: {}, Max rho: {}, Mean rho: {}".format(min_rho, max_rho, mean_rho))

        return avg_rho


    # rho: array, shape (n_frames, npts) - calculated coarse-grained rho for each grid point, for 
    #         each frame
    # weights: (optional) array, shape (n_frames,) - array of weights for averaging rho - default 
    #         is array of equal weights
    def do_output(self):

        if self.rho is None:
            log.warning("Rho not calculated - run calc_rho first")

            return

        log.info('Writing output to \'{}_norm.pdb\' and \'{}_avg.pdb\' '.format(self.outpdb, self.outpdb))

        ##TODO: Outsource this to a dedicated data-manager
        #embed()

        if self.outpdb:
            self.univ.trajectory[0]
            solute_atoms = self.univ.select_atoms(self.mol_sel_spec)
            solute_atoms.atoms.tempfactors = self.avg_rho 
            solute_atoms.atoms.write('{}_norm.pdb'.format(self.outpdb))
            solute_atoms.tempfactors = self.rho_water.mean(axis=0)
            solute_atoms.write('{}_water_avg.pdb'.format(self.outpdb))
            solute_atoms.tempfactors = self.rho_water.var(axis=0)
            solute_atoms.write('{}_water_var.pdb'.format(self.outpdb))
            solute_atoms.tempfactors = self.rho_water.var(axis=0) / self.rho_water.mean(axis=0)
            solute_atoms.write('{}_water_var_norm.pdb'.format(self.outpdb))
            solute_atoms.tempfactors = self.rho_solute.mean(axis=0)
            solute_atoms.write('{}_solute_avg.pdb'.format(self.outpdb))
            

            if (solute_atoms.tempfactors == 0).sum() > 0:
                solute_atoms[solute_atoms.tempfactors==0].write('zero_density.pdb')

            ## Dump all data for each atom with each frame
            np.savez_compressed('rho_data_dump_rad_{}.dat'.format(self.r_cutoff), rho_water=self.rho_water, rho_solute=self.rho_solute, 
                                header='start_frame: {}   end_frame: {}   n_frames: {}    n_solute_atoms: {}'.format(self.start_frame, self.last_frame, self.n_frames, self.n_solute_atoms))
            np.savez_compressed('min_dist_neighbor.dat', min_dist=self.min_dist,
                                header='start_frame: {}   end_frame: {}   sol_spec: {}   neigh_spec: {}'.format(self.start_frame, self.last_frame, self.mol_sel_spec, self.neighbor_sel_spec))

        if self.outxtc:
            print("xtc output not yet supported")


    def go(self):

        #self.setup_grid()
        # Split up frames, assign to work manager, splice back together into
        #   total rho array
        self.calc_rho()
        #embed()
        self.do_output()


if __name__=='__main__':
    DynamicInterface().main()


