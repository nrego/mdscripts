#Instantaneous interface
# nrego sept 2015

from __future__ import division; __metaclass__ = type
import sys
import numpy
import argparse
import logging

from scipy.spatial import cKDTree

import MDAnalysis

from mdtools import Tool

log = logging.getLogger('mdtools.hyd_shell')


class HydrationShell(Tool):
    prog='hydration shell'
    description = '''\
Write out file of all protein heavy atoms - all resnames are replaced by 'SHEL' - for 
easy viewing in VMD (e.g. by set sel [atomselect top "resname SHEL"]; $sel set radius 6.0;)


-----------------------------------------------------------------------------
Command-line options
-----------------------------------------------------------------------------
'''
    
    def __init__(self):
        super(HydrationShell,self).__init__()
        

        self.univ = None

        self.dgrid = None
        self.ngrids = None
        self.gridpts = None
        self.npts = None

        self.output_filename = None

    @property
    def n_frames(self):
        return self.last_frame - self.start_frame
        
    
    def add_args(self, parser):
        
        sgroup = parser.add_argument_group('Hydration Shell options')
        sgroup.add_argument('-f', '--grofile', metavar='INPUT', type=str, required=True,
                            help='Input structure file')
        agroup = parser.add_argument_group('other options')
        agroup.add_argument('-o', '--outfile', default='shell.gro',
                        help='Output file to write hydration shell (default: shell.gro)')


    def process_args(self, args):

        try:
            self.univ = MDAnalysis.Universe(args.grofile, args.grofile)
        except:
            print "Error processing input files: {} and {}".format(args.grofile, args.trajfile)
            sys.exit()

        self.npts = 0

        self.output_filename = args.outfile

    def go(self):

        prot = self.univ.select_atoms('resname LIG and not name H*')

        #non_polar_str = 'resname ALA or resname ILE or resname LEU or resname MET or resname PHE or resname TYR or resname TRP or resname PRO or resname GLY or resname VAL'
        #polar_str = 'protein and not (' + non_polar_str + ')'

        #non_polar = self.univ.select_atoms(non_polar_str)
        #polar = self.univ.select_atoms(polar_str)

        for atom in prot:
            atom.resname = 'SHEL'
        prot.write(self.output_filename)
        #for atom in non_polar:
        #    atom.resname = 'NOP'
        #non_polar.write('non_polar.gro')
        #for atom in polar:
        #    atom.resname = 'POL'
        #polar.write('polar.gro')

if __name__=='__main__':
    HydrationShell().main()


