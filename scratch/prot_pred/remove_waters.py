import numpy as np
from scipy.spatial import cKDTree, ConvexHull, Delaunay

from mdtools import dr

from constants import k

import MDAnalysis

import argparse
from IPython import embed

beta = 1/(300*k)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Remove internal waters from solvated protein')
    parser.add_argument('-c', '--struct', type=str,
                        help='GRO structure file')
    parser.add_argument('--sel-spec', type=str, default='protein',
                        help='Selection spec for selecting all protein atoms (including hydrogens) \
                             default: %(default)s')
    parser.add_argument('--top', '-p', type=str, default='topol.top',
                        help='Topology file (default: %(default)s)')

    args = parser.parse_args()


    univ = MDAnalysis.Universe(args.struct)
    prot = univ.select_atoms(args.sel_spec)

    waters = univ.select_atoms('name OW and not ({})'.format(args.sel_spec))

    hull = Delaunay(prot.positions)

    #positions of waters close to protein (or in it)
    inside_mask = hull.find_simplex(waters.positions) > -1

    print("Removing {} waters".format(inside_mask.sum()))

    univ.add_TopologyAttr('tempfactors')

    waters_to_remove = waters[inside_mask]

    for water in waters_to_remove:
        water.residue.atoms.tempfactors = -1

    atoms = univ.atoms[univ.atoms.tempfactors>-1]
    atoms.write('prot.gro')

    n_waters = atoms.select_atoms('name OW and not ({})'.format(args.sel_spec)).n_atoms

    print('New n waters: {}'.format(n_waters))

