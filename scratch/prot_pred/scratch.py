from __future__ import print_function, division; __metaclass__ = type

import numpy as np
import MDAnalysis

import os, glob
from mdtools import dr
from IPython import embed

from system import MDSystem

fnames = sorted(glob.glob('phi_*/rho_data_dump_rad_6.0.dat.npz'))

dat_0 = np.load('equil/rho_data_dump_rad_6.0.dat.npz')['rho_water'].mean(axis=0)

#sys = MDSystem('equil/npt.tpr', 'equil/npt.gro')
sys = MDSystem('equil/top.tpr', 'equil/cent.gro')
sys.find_buried(dat_0, nb=5)

surf_mask = sys.surf_mask_h
prot_h = sys.prot_h
surf_atoms = prot_h[surf_mask]

global_indices = np.arange(sys.n_prot_tot)
# Gives global index of protein heavy atom, given its local index
heavy_local_to_global = prot_h.indices
local_indices = np.arange(sys.n_prot_h_tot)
# array of local (prot heavy) atoms that have already dewetted
dewet_indices = np.array([])

idx_order = np.array([])

for fname in fnames:
    dirname = os.path.dirname(fname)

    dat = np.load(fname)['rho_water'].mean(axis=0)
    if dat.size > dat_0.size:
        dat = dat[prot_h.indices]

    rho = dat/dat_0
    # Indices of heavy atoms that are dewet and surface atoms
    surf_dewet_mask = (rho < 0.5) & surf_mask
    surf_dewet_idx = local_indices[surf_dewet_mask]
    
    new_dewet_indices = np.setdiff1d(surf_dewet_idx, dewet_indices)
    # Sort by rho
    order_sort = np.argsort(rho[new_dewet_indices])
    idx_order = np.append(idx_order, new_dewet_indices[order_sort])

    dewet_indices = np.unique(np.append(new_dewet_indices, dewet_indices))

idx_order = idx_order.astype(int)

#prot_h.write('order.pdb')
with MDAnalysis.Writer('order.pdb', multiframe=True, bonds=None, n_atoms=prot_h.n_atoms) as PDB:

    for i, idx in enumerate(idx_order):
        prot_h[idx].tempfactor = 1
        prot_h[idx].name = 'D'
        PDB.write(prot_h.atoms)
        prot_h.write('pdb_{}.pdb'.format(i))


