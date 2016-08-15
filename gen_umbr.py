'''
Quick script to construct dyn_union_sph_sh umbrella file around .gro file's heavy atoms
'''

from itertools import izip
import sys

import numpy
from scipy.spatial import cKDTree

from MDAnalysis import Universe

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Generate dynamic umbrella conf file from Gro structure")
    parser.add_argument("-c", "--structure", required=True, dest="gro",
                        help="gromacs structure '.gro' input file")
    parser.add_argument("-f", "--traj",
                        help="trajectory file (optional) - if supplied, output heavy atoms with more than avg" \
                        "number of waters within radius")
    parser.add_argument("-b", "--start", default=0, type=float,
                        help="Starting time of trajectory, in ps (default 0 ps)")
    parser.add_argument("-d", "--waterdist", default=5.5, type=float,
                        help="Radius over which to search for waters, if desired")
    parser.add_argument("-aw", "--avgwater", default=1.0, type=float,
                        help="Minimum average number of waters within radius of heavy atom in order to print" \
                        "out atom to umbr file")
    parser.add_argument("-r", "--radius", dest="rad", default=6, type=float,
                        help="radius of hydration shell around heavy atoms (default 6 A)")
    parser.add_argument("--static", action='store_true',
                        help="If true, generate conf file for STATIC INDUS (default: false)")
    parser.add_argument("-o", "--out", default="umbr.conf", dest="outfile",
                        help="output file name (default: umbr.conf)")
    parser.add_argument("--sspec", type=str,
                        help="custom selection string (to select atoms over which to create umbrellas), default all prot heavies")

    args = parser.parse_args()

    if not args.static:
        header_string = "; Umbrella potential for a spherical shell cavity\n"\
        "; Name    Type          Group  Kappa   Nstar    mu    width  cutoff  outfile    nstout\n"\
        "hydshell dyn_union_sph_sh   OW  0.0     0   XXX    0.01   0.02   phiout.dat   50  \\\n"
    else:
        header_string = "; Umbrella potential for a spherical shell cavity\n"\
        "; Name    Type          Group  Kappa   Nstar    mu    width  cutoff  outfile    nstout\n"\
        "hydshell union_sph_sh   OW  0.0     0   XXX    0.01   0.02   phiout.dat   50  \\\n"        

    if args.traj is None:
        u = Universe(args.gro)

        if args.sspec is not None:
            prot_heavies = u.select_atoms(args.sspec)
        else:
            # Select peptide heavies - exclude water's and ions
            prot_heavies = u.select_atoms("not (name H* or type H or resname SOL) and not (name NA or name CL) and not (resname WAL) and not (resname DUM)")

        fout = open(args.outfile, 'w')
        fout.write(header_string)

        if args.static:
            for atm in prot_heavies:
                fout.write("{:<10.1f} {:<10.1f} {:<10.3f} {:<10.3f} {:<10.3f}\\\n".format(-0.5, args.rad/10.0, atm.pos[0], atm.pos[1], atm.pos[2]))
        else:
            for atm in prot_heavies:
                fout.write("{:<10.1f} {:<10.1f} {:d} \\\n".format(-0.5, args.rad/10.0, atm.index+1))

        fout.close()

    # Only print out heavy atoms near water

    else:
        u = Universe(args.gro, args.traj)

        frame_time = u.trajectory.totaltime / u.trajectory.n_frames
        startframe = int(args.start/frame_time)
        lastframe = u.trajectory.n_frames

        waterdist = args.waterdist

        n_frames = lastframe - startframe

        prot_heavies = u.select_atoms("not (name H* or type H or resname SOL) and not (name NA or name CL) and not (resname WAL) and not (resname DUM)")
        water_ow = u.select_atoms("name OW")

        # Number of waters near each protein heavy
        n_waters = numpy.zeros((len(prot_heavies),), dtype=numpy.int32)


        for frame_idx in xrange(startframe, lastframe):
            print "Frame: {}\r".format(frame_idx),
            sys.stdout.flush()

            u.trajectory[frame_idx]

            # KD tree of water O positions
            tree = cKDTree(water_ow.positions)

            for prot_idx, prot_atom in enumerate(prot_heavies):
                water_neighbors = tree.query_ball_point(prot_atom.position, waterdist)
                n_waters[prot_idx] += len(water_neighbors)

        n_waters /= n_frames
        print "\n"

        with open(args.outfile, 'w') as fout:
            fout.write(header_string)

            # n_waters is array of shape (n_prot_heavies, ) that has the number of nearest waters for each prot heavy atom
            if args.static:
                for atomidx in xrange(n_waters.shape[0]):
                    if n_waters[atomidx] > args.avgwater:
                        atm = prot_heavies[atomidx]
                        fout.write("{:<10.1f} {:<10.1f} {:<10.3f} {:<10.3f} {:<10.3f}\\\n".format(-0.5, args.rad/10.0, atm.pos[0]/10.0, atm.pos[1]/10.0, atm.pos[2]/10.0))                   
            else:
                for atomidx in xrange(n_waters.shape[0]):
                    if n_waters[atomidx] > args.avgwater:
                        atm = prot_heavies[atomidx]
                        fout.write("{:<10.1f} {:<10.1f} {:d} \\\n".format(-0.5, args.rad/10.0, atm.index+1))


