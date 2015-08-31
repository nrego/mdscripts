'''
Quick script to construct dyn_union_sph_sh umbrella file around .gro file's heavy atoms
'''

from itertools import izip

import numpy

from MDAnalysis import Universe

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Generate dynamic umbrella conf file from Gro structure")
    parser.add_argument("-c", "--structure", required=True, dest="gro",
                        help="gromacs structure '.gro' input file")
    parser.add_argument("-r", "--radius", dest="rad", default=6, type=float,
                        help="radius of hydration shell around heavy atoms (default 6 A)")
    parser.add_argument("-o", "--out", default="output", dest="outfile",
                        help="output")

    args = parser.parse_args()

    universe = Universe(args.gro)

    # Select peptide heavies - exclude water's and ions
    heavyAtms = universe.selectAtoms("not (name H* or resname SOL) and not name NA")

    indices = heavyAtms.indices()
    # Consistent with gromacs numbering
    indices += 1

    headerString = "; Umbrella potential for a spherical shell cavity\n\
; Name    Type          Group  Kappa   Nstar    mu    width  cutoff  outfile    nstout\n\
CAVITATOR dyn_union_sph_sh   OW  0.0     0   XXX    0.01   0.02   phiout.dat   50  \\\n"

    fout = open(args.outfile, 'w')
    fout.write(headerString)

    for i in indices:
        fout.write("{:<10.1f} {:<10.1f} {:d} \\\n".format(-0.5, args.rad/10.0, i))

    fout.close()
