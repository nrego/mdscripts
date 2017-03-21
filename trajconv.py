from __future__ import print_function, division; __metaclass__ = type

import numpy as np
import logging

import MDAnalysis

from boxutils import center_mol, rotate_mol

from selection_specs import sel_spec_heavies_nowall

from mdtools import ParallelTool

import sys

class Trajconv(ParallelTool):
    prog='trajconv'
    description = '''\
Center and align a structure (PDB or GRO) or trajectory (XTC or TRR) to a reference
structure (Requires a GRO and a TPR file)

Automatically treats PBC for selected molecule group, assuring molecule to be centered
  and aligned is whole in each frame. 

  NOTE: **This tool assumes the reference structure is whole** and will not work correctly
    otherwise (it will throw an error if it finds the reference structure is broken)
  The reference structure will automaticall be centered according to its COM before any
    fitting or alignment


-----------------------------------------------------------------------------
Command-line options
-----------------------------------------------------------------------------
'''
    
    def __init__(self):
        super(Trajconv,self).__init__()
        
        # Parallel processing by default (this is not actually necessary, but it is
        # informative!)
        self.wm_env.default_work_manager = self.wm_env.default_parallel_work_manager

        self.ref_univ = None
        self.other_univ = None
        self.outputfilename = None

        # are we processing a trajectory?
        self.do_traj = False

        self.start_frame = None
        self.end_frame = None

        self.sel_spec = None

        self.rmsd_out = None

    @property
    def n_frames(self):
        return self.last_frame - self.start_frame

        
    def add_args(self, parser):
        
        sgroup = parser.add_argument_group('Trajconv options')
        sgroup.add_argument('-s1', '--tprfile1', metavar='TPR', type=str, required=True,
                            help='Input topology file (tpr) for ref structure')
        sgroup.add_argument('-s2', '--tprfile2', metavar='TPR', type=str, required=True,
                            help='Input topology file (tpr) for structure/trajectory to fit')
        sgroup.add_argument('-c', '--grofile', metavar='GRO', type=str, required=True,
                            help='Input reference structure file')
        sgroup.add_argument('-f', '--fitfile', metavar='XTC', type=str, required=True,
                            help='Input file to fit to reference. can be GRO or XTC')
        sgroup.add_argument('-b', '--start', type=int, default=0,
                            help='First timepoint (in ps)')
        sgroup.add_argument('-e', '--end', type=int, 
                            help='Last timepoint (in ps) - default is last available')
        sgroup.add_argument('--fitspec', type=str, default=sel_spec_heavies_nowall,
                            help='MDAnalysis selection string for fitting. Default selects all protein heavy atoms')

        agroup = parser.add_argument_group('other options')
        agroup.add_argument('-o', '--outfile', type=str, default='fit.gro',
                        help='Output file to write fitted trajectory or structure. File type determined by input')
        agroup.add_argument('-orms', '--outrmsd', type=str, default='rmsd_fit.dat',
                        help='Output rmsd values for each frame after fitting')

    def process_args(self, args):

        #try:
        self.ref_univ = MDAnalysis.Universe(args.tprfile1, args.grofile)
        ext = args.fitfile.split('.')[-1]
        if ext in ['trr', 'xtc']:
            self.do_traj = True
            self.other_univ = other_univ = MDAnalysis.Universe(args.tprfile2, args.fitfile)
        elif ext == 'gro':
            self.other_univ = other_univ = MDAnalysis.Universe(args.tprfile2, args.fitfile)
        else:
            print("unknown or missing extension")
            sys.exit()
        #except:
        #    print("Error processing input files: {} and {}".format(args.grofile, args.fitfile))
        #    sys.exit()


        if (args.start > (other_univ.trajectory.n_frames * other_univ.trajectory.dt)):
            raise ValueError("Error: provided start time ({} ps) is greater than total time ({} ps)"
                             .format(args.start, (other_univ.trajectory.n_frames * other_univ.trajectory.dt)))

        self.start_frame = int(args.start / other_univ.trajectory.dt)
        if args.end is not None:
            self.last_frame = args.end
        else:
            self.last_frame = other_univ.trajectory.n_frames

        self.sel_spec = args.fitspec

        self.outfile = args.outfile.split('.')[0]
        self.rmsd_out = args.outrmsd

    def go(self):

        n_frames = self.last_frame - self.start_frame
        center_mol(self.ref_univ)
        rmsd = np.zeros((self.last_frame-self.start_frame))
        self.ref_univ.atoms.write('fit_ref.gro')
        if self.do_traj:
            with MDAnalysis.Writer(self.outfile + ".xtc", self.other_univ.atoms.n_atoms) as W:
                for i_frame in range(self.start_frame, self.last_frame):
                    if i_frame % 100 == 0:
                        print("\r doing frame {} of {}".format(i_frame, self.last_frame))
                    curr_ts = self.other_univ.trajectory[i_frame]

                    center_mol(self.other_univ, do_pbc=True)
                    rms = rotate_mol(self.ref_univ, self.other_univ, mol_spec=self.sel_spec)
                    if i_frame == self.start_frame:
                        self.other_univ.atoms.write('first_frame_fit.gro')
                    W.write(self.other_univ.atoms)
                    rmsd[i_frame-self.start_frame] = rms
            np.savetxt(self.rmsd_out, rmsd)

        else:
            center_mol(self.other_univ, do_pbc=True)
            rms = rotate_mol(self.ref_univ, self.other_univ, mol_spec=self.sel_spec)
            self.other_univ.atoms.write(self.outfile + ".gro")




if __name__=='__main__':
    Trajconv().main()