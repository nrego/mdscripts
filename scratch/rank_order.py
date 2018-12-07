from __future__ import division, print_function

import numpy as np
import MDAnalysis
from mdtools import MDSystem

import os, glob
from IPython import embed

import argparse

parser = argparse.ArgumentParser('Rank-order heavy atoms by dewetting order')
parser.add_argument('fnames', type=str, nargs='+', metavar='PHI',
                    help='names of all rho_data_dump (phiout.dat assumed to be in same folder)')
parser.add_argument('-s', '--top', type=str, default='top.tpr',
                    help='TPR file')
parser.add_argument('-c', '--struct', type=str, required=True,
                    help='Strcutre file')
parser.add_argument('--sel-spec', type=str, default='segid A',
                    help='selspec for mdsystem')

args = parser.parse_args()
## Hardcoded params ##
#fnames = glob.glob('phi_*/rho_data_dump.dat.npz')
fnames = args.fnames

phi_vals = np.array([float(fname.split('/')[0].split('_')[-1]) / 10.0 for fname in fnames])
n_phi = phi_vals.size
assert np.array_equal(np.arange(n_phi), np.argsort(phi_vals))



f0 = fnames[0]
assert f0.split('/')[0] == 'phi_000'
dat_0 = np.load(f0)['rho_water'].mean(axis=0)

top = args.top
#struct = 'equil_bulk_cent.gro'
struct = args.struct

buried_thresh = 5

## End hardcoded params ##

sys = MDSystem(top, struct, sel_spec=args.sel_spec)

sys.find_buried(dat_0, nb=buried_thresh)

# Shape: (n_heavy_atoms,) [(487,)]
surf_mask = sys.surf_mask_h 
prot = sys.prot_h
print('N heavy atoms: {}'.format(sys.n_prot_h_tot))
print('N surface heavies: {}'.format(sys.n_surf_h))

phi_vals = []
last_phi = -1

# Indices from (0...n_surf_h-1)
# Indices 0-indexed according to heavy atoms!!
heavy_indices = np.arange(sys.n_prot_h_tot)

## Indices, in order, of heavy atoms that have dewetted.
#    These are heavy atom indices (from 0 to n_heavies-1), but will not include
#    any buried atoms
dewet_indices = np.array([], dtype=int)

for fname in fnames:

    phi = float(fname.split('/')[0].split('_')[-1]) / 10.0

    assert phi > last_phi
    last_phi = phi

    phi_vals.append(phi)

    this_dat = np.load(fname)['rho_water'].mean(axis=0)

    this_rho = this_dat / dat_0

    # Shape (n_heavies,)
    #  True if a surface atom and dewetted at this phi
    this_dewet_mask = (this_rho < 0.5) & surf_mask

    # Indices of all heavy atoms that are on the surface and dewetted at this phi
    this_dewet_indices = heavy_indices[this_dewet_mask]

    print('  phi: {}'.format(phi))
    print('  n surf heavies dewet: {}'.format(this_dewet_mask.sum()))

    # Indices of heavy surface atoms that have dewet for the first time at this phi
    new_dewet_indices = np.setdiff1d(this_dewet_indices, dewet_indices)

    # Now sort them by their rho vals...
    sort_idx = np.argsort(this_rho[new_dewet_indices])

    # ...And add the (sorted by rho) indices to the list
    dewet_indices = np.append(dewet_indices, new_dewet_indices[sort_idx])



## Now color the (heavy, surface) atoms by the order in which they dewetted
for i, idx in enumerate(dewet_indices):

    prot[idx].tempfactor = i / 10.0

prot.write('ordered.pdb', bonds=None)

np.savetxt('dewet_indices.dat', dewet_indices, fmt='%1d')
np.savetxt('surf_mask.dat', surf_mask, fmt='%1d')

