from __future__ import division, print_function

import numpy as np
import MDAnalysis
from mdtools import MDSystem, dr

import os, glob

top = 'top.tpr'
struct = 'equil_bulk_cent.gro'

surf_mask = np.loadtxt('surf_mask.dat', dtype=bool)
dewet_indices = np.loadtxt('dewet_indices.dat', dtype=int)

clust_size = 10
first_clust_indices = dewet_indices[:clust_size]

fnames = glob.glob('../hfb_traj/phi_*/traj_cent.xtc')

last_phi = -1
phi_vals = []
n_start = 1000

u0 = MDAnalysis.Universe(top, fnames[0])
prot = u0.select_atoms('protein and not name H*')
global_first_clust_indices = prot[first_clust_indices].indices

## Put together our selection string...
sel_str = 'bynum {:d}'.format(global_first_clust_indices[0]+1)
for i in range(1,clust_size):
    sel_str += ' or bynum {:d}'.format(global_first_clust_indices[i]+1)

ds0 = dr.loadPhi('phi_000/phiout.dat')
ts = ds0.ts
skip = int(1/ts)
print('N frames: {}'.format(u0.trajectory.n_frames))
print('start frame: {}'.format(n_start))

n_clust = np.zeros((len(fnames), u0.trajectory.n_frames-n_start))
n_tot = np.zeros_like(n_clust)

for i, fname in enumerate(fnames):

    this_phi = float(fname.split('/')[2].split('_')[-1]) / 10.0
    print("Phi: {}".format(this_phi))
    assert this_phi > last_phi
    last_phi = this_phi
    phi_vals.append(this_phi)

    u = MDAnalysis.Universe(top, fname)

    prot = u.select_atoms('segid seg_0_Protein_targ and not name H*')
    water = u.select_atoms('name OW')

    this_phi_ds = dr.loadPhi('phi_{:03d}/phiout.dat'.format(int(10*this_phi)))
    assert this_phi_ds.phi == this_phi

    n_tot[i,:] = this_phi_ds.data[n_start::skip]['$\~N$']

    for i_frame in range(n_start, u.trajectory.n_frames):
        if i_frame % 1000 == 0:
            print("  frame: {}".format(i_frame))
        u.trajectory[i_frame]

        # Atom group of waters within 6 A of any atom in clust, for this frame and phi
        this_frame_clust_waters = u.select_atoms('name OW and around 6 ({})'.format(sel_str))

        n_clust[i, i_frame-n_start] = this_frame_clust_waters.n_atoms


