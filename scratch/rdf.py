# Nbr 05/30/19
# (use %paste in IPython)

from __future__ import division, print_function

import numpy as np
import MDAnalysis
from scipy.spatial import cKDTree

def pbc(atoms, i_dim, box_max, box_min):
    box_vec = box_max - box_min
    over_dim = atoms.positions[:,i_dim] > box_max[i_dim]
    under_dim = atoms.positions[:,i_dim] < box_min[i_dim]

    shift = np.zeros(3)
    shift[i_dim] = box_vec[i_dim]

    if over_dim.sum() > 0:
        atoms[over_dim].positions -= shift
    if under_dim.sum() > 0:
        atoms[under_dim].positions += shift

    return atoms

def center_by_water(atoms, probe_atom_idx):
    box_min = atoms.positions.min(axis=0)
    atoms.positions -= box_min
    box_min = atoms.positions.min(axis=0)
    box_max = atoms.positions.max(axis=0)

    box_cent = 0.5*(box_max - box_min)

    probe_atom_pos = atoms[probe_atom_idx].position
    # Center on probe atom
    shift = box_cent - probe_atom_pos
    atoms.positions += shift

    ## Apply pbc's to anything that's outside the box
    pbc(atoms, 0, box_max, box_min)
    pbc(atoms, 1, box_max, box_min)
    pbc(atoms, 2, box_max, box_min)

    return atoms, box_min, box_max

univ = MDAnalysis.Universe('equil.gro', 'traj_comp.xtc')
water_ow = univ.select_atoms('name OW')

dr = 0.1
max_r = 10.0
r_bins = np.arange(0, max_r+(2*dr), dr)
r_vals = r_bins[:-1]
sphere_shell_vols = 4*np.pi*r_vals**2*dr

n_frames = univ.trajectory.n_frames
n_waters = water_ow.n_atoms
n_rvals = r_bins.shape[0] - 1

# shape: (n_frames x n_ow x n_rbins-1)
r_occ = np.zeros((n_frames, n_waters, n_rvals))

for i_frame, ts in enumerate(univ.trajectory):
    print("frame: {:d}".format(i_frame))
    for i_water in range(n_waters):
        univ.trajectory[i_frame]
        centered_waters, box_min, box_max = center_by_water(water_ow, i_water)
        assert np.array_equal(box_min, np.zeros(3))

        tree = cKDTree(centered_waters.positions)

        dists, neigh = tree.query(centered_waters[i_water].position, k=n_waters, distance_upper_bound=max_r)
        dists[i_water] = np.inf # don't count itself
        masked_dists = dists[dists < np.inf]

        # Now find which bin each neighboring atom belongs to
        bin_mask = np.digitize(masked_dists, r_bins) - 1

        # Ugh!
        for i_bin in bin_mask:
            r_occ[i_frame, i_water, i_bin] += 1 

        r_occ[i_frame, i_water] /= sphere_shell_vols


g_of_r = r_occ.mean(axis=(0,1))


