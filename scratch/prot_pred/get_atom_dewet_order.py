from __future__ import print_function, division; __metaclass__ = type

import numpy as np
import MDAnalysis

import os, glob
from mdtools import dr
from IPython import embed

from mdtools.system import MDSystem

import argparse

parser = argparse.ArgumentParser('Collect atom dewetting info from phi-ensemble simulation, order atoms')
parser.add_argument('--ref-top', required=True, type=str,
                    help='Reference topology')
parser.add_argument('--ref-struct', required=True, type=str,
                    help='Reference structure')
parser.add_argument('--sel-spec', default='segid targ', type=str,
                    help='Selection spec for reference protein (default: "segid targ")')
parser.add_argument('--ref-rho', required=True, type=str,
                    help='Reference rho for determining buried atoms')
parser.add_argument('-nb', default=5, type=float,
                    help='Considered buried if fewer than this number of waters in reference struct (default 5)')
parser.add_argument('--infiles', type=str, nargs='+',
                    help='input files (e.g. phi_*)')
args = parser.parse_args()




fnames = np.sort(args.infiles)
ref_rho = np.load(args.ref_rho)['rho_water'].mean(axis=0)
sys = MDSystem(args.ref_top, args.ref_struct, sel_spec=args.sel_spec)

sys.find_buried(ref_rho, nb=5)
#embed()
surf_mask = sys.surf_mask_h
prot_h = sys.prot_h
surf_atoms = prot_h[surf_mask]
surf_atoms.tempfactors = -1

global_indices = np.arange(sys.n_prot_tot)
# Gives global index of protein heavy atom, given its local index
heavy_local_to_global = prot_h.indices
local_indices = np.arange(sys.n_prot_h_tot)
# array of local (prot heavy) atoms that have already dewetted
dewet_indices = np.array([])

idx_order = np.array([])

prev_phi = -1
for fname in fnames:
    dirname = os.path.dirname(fname)
    phi = float(dirname.split('_')[-1]) / 10.0
    assert phi > prev_phi
    prev_phi = phi

    dat = np.load(fname)['rho_water'].mean(axis=0)
    if dat.size > ref_rho.size:
        dat = dat[prot_h.indices]

    rho = dat/ref_rho
    # Indices of heavy atoms that are dewet and surface atoms
    surf_dewet_mask = (rho < 0.5) & surf_mask
    surf_dewet_idx = local_indices[surf_dewet_mask]
    
    new_dewet_indices = np.setdiff1d(surf_dewet_idx, idx_order)
    # Sort by rho
    order_sort = np.argsort(rho[new_dewet_indices])
    idx_order = np.append(idx_order, new_dewet_indices[order_sort])

        
idx_order = idx_order.astype(int)
print("{} total atoms dewetted".format(len(idx_order)))
print("limiting to the first 500 atoms")
max_i = min(len(idx_order), 500)
#prot_h.write('order.pdb')
with MDAnalysis.Writer('order.pdb', multiframe=True, bonds=None, n_atoms=prot_h.n_atoms) as PDB:

    for i in range(max_i):
        idx = idx_order[i]
        prot_h[idx].tempfactor = i/10.0
        prot_h[idx].name = 'D'
        PDB.write(prot_h.atoms)
        prot_h.write('pdb_{}.pdb'.format(i))


