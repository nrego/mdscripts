
import numpy as np
import logging

import MDAnalysis

from boxutils import center_mol, rotate_mol

from constants import SEL_SPEC_HEAVIES_NOWALL

from mdtools import ParallelTool

from MDAnalysis.analysis.rms import rmsd

import sys

from IPython import embed

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
        self.rmsd_spec = None

        self.sel_spec_other = None
        self.rmsd_spec_other = None

        self.rmsd_out = None

        self.center_only = None

        # Shape: (n_frames, n_rms_specs+1)
        self.rmsd_arr = None
        # rms per-atom (for each frame) for rms on the rmsdspec (empty if none supplied)
        self.rms_per_atom = None

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
        sgroup.add_argument('--fitspec', type=str, default=SEL_SPEC_HEAVIES_NOWALL,
                            help='MDAnalysis selection string for fitting. Default selects all protein heavy atoms')
        sgroup.add_argument('--fitspec-other', type=str,
                            help='Fit spec for selecting the other structure to fit (default: same as fitspec)')
        sgroup.add_argument('--center-only', action='store_true', 
                            help='If true, only center molecule (no fitting)')
        sgroup.add_argument('--rmsdspec', type=str, 
                            help='MDAnalysis selection string for rmsd (after fitting on fitspec). Optional.')
        sgroup.add_argument('--rmsdspec-other', type=str,
                            help='Sel spec for other structure rmsd (default: same as rmsdspec)')
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
        elif ext == 'gro' or ext == 'pdb':
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
        self.rmsd_spec = args.rmsdspec

        self.sel_spec_other = args.fitspec_other or args.fitspec
        self.rmsd_spec_other = args.rmsdspec_other or args.rmsdspec

        self.outfile = args.outfile.split('.')[0]
        self.rmsd_out = args.outrmsd

        self.center_only = args.center_only



    def go(self):

        header_str = "fitspec: {}; rmsdspec: {};  fitspec_other: {};  rmsd_spec_other: {}".format(self.sel_spec, self.rmsd_spec, self.sel_spec_other, self.rmsd_spec_other)

        n_frames = self.last_frame - self.start_frame

        ndim = 2 if self.rmsd_spec is None else 3
        self.rmsd_arr = np.zeros((self.n_frames, ndim))

        self.ref_univ.atoms.write('fit_ref.gro')
        
        if self.rmsd_spec is not None:
            ref_struct = self.ref_univ.select_atoms(self.rmsd_spec)
            other_struct = self.other_univ.select_atoms(self.rmsd_spec_other)
            
            assert ref_struct.n_atoms == other_struct.n_atoms
            
            self.rms_per_atom = np.zeros((self.n_frames, ref_struct.n_atoms))

        if self.do_traj:
            with MDAnalysis.Writer(self.outfile + ".xtc", self.other_univ.atoms.n_atoms) as W:
                for i_frame in range(self.start_frame, self.last_frame):
                    if i_frame % 100 == 0:
                        print("doing frame {} of {}".format(i_frame, self.last_frame))
                        sys.stdout.flush()
                    curr_ts = self.other_univ.trajectory[i_frame]

                    center_mol(self.other_univ, do_pbc=False)
                    if not self.center_only:
                        rms = rotate_mol(self.ref_univ, self.other_univ, ref_spec=self.sel_spec, other_spec=self.sel_spec_other)
                        self.rmsd_arr[i_frame-self.start_frame, 0] = curr_ts.time
                        self.rmsd_arr[i_frame-self.start_frame, 1] = rms          

                    if i_frame == self.start_frame:
                        self.other_univ.atoms.write('first_frame_fit.gro')
                    W.write(self.other_univ.atoms)


                    if self.rmsd_spec is not None and not self.center_only:
                        rms_other = rmsd(ref_struct.atoms.positions, other_struct.atoms.positions)
                        self.rmsd_arr[i_frame-self.start_frame, 2] = rms_other

                        self.rms_per_atom[i_frame-self.start_frame, :] = np.sqrt( np.sum((ref_struct.atoms.positions - other_struct.atoms.positions)**2, axis=1) )


        else:
            center_mol(self.other_univ, do_pbc=False, check_broken=False)
            rms = rotate_mol(self.ref_univ, self.other_univ, ref_spec=self.sel_spec, other_spec=self.sel_spec_other)
            self.other_univ.atoms.write(self.outfile + ".gro")

            self.rmsd_arr[0,0] = 0.0
            self.rmsd_arr[0,1] = rms
            if self.rmsd_spec is not None:
                rms_other = rmsd(ref_struct.atoms.positions, other_struct.atoms.positions)
                self.rmsd_arr[0,2] = rms_other

                self.rms_per_atom[0,:] = np.sqrt( np.sum((ref_struct.atoms.positions - other_struct.atoms.positions)**2, axis=1) )

        if self.rmsd_spec is not None:
            avg_rms_per_atom = self.rms_per_atom.mean(axis=0)
            self.other_univ.add_TopologyAttr('tempfactors')
            other_struct.tempfactors = avg_rms_per_atom
            other_struct.write('fit_per_atom_rmsd.pdb', bonds=None)

            self.ref_univ.add_TopologyAttr('tempfactors')
            ref_struct.tempfactors = avg_rms_per_atom
            ref_struct.write('fit_ref_per_atom_rmsd.pdb', bonds=None)

        # Save output
        np.savetxt(self.rmsd_out, self.rmsd_arr, header=header_str)
        np.savez_compressed('rms_per_atom.dat', header=self.rmsd_spec, rms_per_atom=self.rms_per_atom)


if __name__=='__main__':
    Trajconv().main()
