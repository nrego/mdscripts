from __future__ import division, print_function

import numpy as np
import MDAnalysis
from mdtools import MDSystem, dr

import os, glob

import argparse

from IPython import embed

## Put together our selection string...
def build_sel_str(global_indices):
    sel_str = 'bynum {:d}'.format(global_indices[0]+1)
    for i in range(1,clust_size):
        sel_str += ' or bynum {:d}'.format(global_indices[i]+1)

    return sel_str


parser = argparse.ArgumentParser('Rank-order heavy atoms by dewetting order')
parser.add_argument('fnames', type=str, nargs='+', metavar='PHI',
                    help='names of all trajectory files (phiout.dat assumed to be in same folder)')
parser.add_argument('-s', '--top', type=str, default='top.tpr',
                    help='TPR file')
parser.add_argument('-b', '--n-start', type=int, default=1000,
                    help='Start frame for analysis')
parser.add_argument('--sel-spec', type=str, default='segid A',
                    help='selspec for mdsystem')
parser.add_argument('--dewet-indices', type=str, default='dewet_indices.dat',
                    help='File with dewetted indices')
parser.add_argument('--surf-mask', type=str, default='surf_mask.dat',
                    help='File with surf mask')
parser.add_argument('--clust-size', type=int, default=10,
                    help='Cluster size')
args = parser.parse_args()

top = args.top

surf_mask = np.loadtxt(args.surf_mask, dtype=bool)
dewet_indices = np.loadtxt(args.dewet_indices, dtype=int)

clust_size = args.clust_size

n_start = args.n_start
sel_spec = args.sel_spec

#fnames = glob.glob('../hfb_traj/phi_*/traj_cent.xtc')
fnames = args.fnames

phi_vals = np.array([float(fname.split('/')[0].split('_')[-1]) / 10.0 for fname in fnames])
n_phi = phi_vals.size
assert np.array_equal(np.arange(n_phi), np.argsort(phi_vals))
assert phi_vals[0] == 0

sys = MDSystem(top, fnames[0], sel_spec=sel_spec)

n_frames = sys.univ.trajectory.n_frames

prot = sys.prot_h
n_dewet_atoms = dewet_indices.size
n_clust = n_dewet_atoms // clust_size
local_indices = np.arange(prot.n_atoms)
global_indices = prot.indices
surf_indices = local_indices[surf_mask]
# Sanity - dewet indices must be subset of surf_indices
assert np.isin(dewet_indices, surf_indices).all()

# Find time step for all data
ds0 = dr.loadPhi('phi_000/phiout.dat')
ts = ds0.ts
skip = int(1/ts)
print('N frames: {}'.format(n_frames))
print('start frame: {}'.format(n_start))
print('{} dewetted atoms'.format(n_dewet_atoms))
print('clust size: {} atoms'.format(clust_size))
print('N clusters: {}'.format(n_clust))

n_subvol = np.zeros((n_clust, len(fnames), n_frames-n_start))
n_tot = np.zeros_like(n_subvol[0])
print('')
print('...gathering total N_v for each phi...')
## Get N_tot for each phi...
for i, fname in enumerate(fnames):

    this_phi = float(fname.split('/')[0].split('_')[-1]) / 10.0
    assert this_phi == phi_vals[i]

    this_phi_ds = dr.loadPhi('phi_{:03d}/phiout.dat'.format(int(10*this_phi)))
    assert this_phi_ds.phi == phi_vals[i]

    n_tot[i,:] = this_phi_ds.data[n_start::skip]['$\~N$']

print('')
print('Gathering n_clust for each phi...')
for i, fname in enumerate(fnames):

    this_phi = phi_vals[i]
    print("Phi: {}".format(this_phi))

    this_sys = MDSystem(top, fname, sel_spec)

    u = this_sys.univ

    for i_frame in range(n_start, u.trajectory.n_frames):
        if i_frame % 1000 == 0:
            print("  frame: {}".format(i_frame))
        u.trajectory[i_frame]

        for i_clust in range(n_clust):
            
            start_idx = i_clust*clust_size
            end_idx = (i_clust+1)*clust_size

            clust_local_indices = dewet_indices[start_idx:end_idx]
            clust_global_indices = global_indices[clust_local_indices]
            sel_str = build_sel_str(clust_global_indices)
            # Atom group of waters within 6 A of any atom in clust, for this frame and phi
            this_frame_clust_waters = u.select_atoms('name OW and around 6 ({})'.format(sel_str))

            n_subvol[i_clust, i, i_frame-n_start] = this_frame_clust_waters.n_atoms

avg_n_tot = n_tot.mean(axis=1)

avg_n_tot_sq = (n_tot**2).mean(axis=1)

var_n = avg_n_tot_sq - avg_n_tot**2
embed()
for i in range(n_clust):

    this_clust_n = n_subvol[i]
    this_clust_avg_n = this_clust_n.mean(axis=1)
    this_cross = this_clust_n * n_tot
    this_avg_cross = this_cross.mean(axis=1)

    this_cov = this_avg_cross - (this_avg_clust * avg_n_tot)



