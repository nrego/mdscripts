
import numpy as np
import MDAnalysis

import argparse
from IPython import embed

import pickle
from matplotlib import pyplot as plt

from scratch.neural_net import *
from scratch.sam.util import *

from scipy.spatial import cKDTree

import os,sys


parser = argparse.ArgumentParser('Measure the number of water (oxygens) w/in distance of patch atoms')
parser.add_argument('-c', '--struct', type=str, default='equil/confout.gro',
                    help='Structure of patterned SAM')
parser.add_argument('-f', '--traj', type=str, default='equil/traj.xtc',
                    help='Structure of patterned SAM')
parser.add_argument('-rc', type=float, default=5.0,
                    help='Radius of union spheres on methyl CT atoms')
parser.add_argument('--ro-vals', type=str,
                    help='numpy array of radii for hydroxyl OH atoms')

args = parser.parse_args()

univ = MDAnalysis.Universe(args.struct, args.traj)

rc = args.rc
ro_vals = eval(args.ro_vals)

print("Rc: {:.2f}".format(rc))
print("Ro vals: {}\n".format(ro_vals))

n_frames = univ.trajectory.n_frames
n_waters = np.zeros((n_frames, ro_vals.size))

n_res = univ.residues.n_residues
sel_spec = "resid {}-{}".format(n_res-35, n_res)

assert univ.select_atoms("{} and (name CT or name O12)".format(sel_spec)).n_atoms == 36

for i_frame, ts in enumerate(univ.trajectory):
    if i_frame % 50 == 0:
        print("frame: {}".format(i_frame+1))
    for i_ro, ro in enumerate(ro_vals):
        sel = univ.select_atoms("(name OW and around {:.2f} name CT) or (name OW and around {:.2f} ({} and name O12))".format(rc, ro, sel_spec))

        n_waters[i_frame, i_ro] = sel.n_atoms


np.savez_compressed("n_waters_{:02d}.dat".format(int(rc*10)), n_waters=n_waters, rc=rc, ro_vals=ro_vals)



