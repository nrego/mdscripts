
import numpy as np
import MDAnalysis

import argparse
from IPython import embed
import os, glob

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np

from scratch.sam.util import *

from scipy.special import binom

def get_p_q(dname):
    splits = dname.split('_')

    return int(splits[1]), int(splits[3])


### Extract all sampled dos into big array
#########################################

#p = q = 6

dnames = sorted(glob.glob('p_*'))
#dnames = ['p_{:02d}_q_{:02d}'.format(p,q)]
n_dir = len(dnames)

print('N directories (p, q combos): {}'.format(n_dir))

# np.unique sorts automatically
vals_pq = np.array([get_p_q(dname) for dname in dnames])
p_vals = np.unique(vals_pq[:,0])
q_vals = np.unique(vals_pq[:,1])

max_ko = vals_pq.prod(axis=1).max()
vals_ko = np.arange(max_ko+1)

# max n_oo and n_oe for each p,q
edge_vals = np.zeros_like(vals_pq)
for i, pq in enumerate(vals_pq):
    state = State(np.array([],dtype=int), p=pq[0], q=pq[1])

    edge_vals[i] = state.n_oo, state.n_oe

max_noo, max_noe = edge_vals.max(axis=0)
vals_noo = np.arange(max_noo+1)
vals_noe = np.arange(max_noe+1)

# Fixed volume - will have to combine to get pressures
x_ko, x_noo, x_noe = np.meshgrid(vals_ko, vals_noo, vals_noe, indexing='ij')

# Shape: (p_q, vals_ko, vals_noo, vals_noe)
dos = np.zeros((vals_pq.shape[0], *x_ko.shape))
entropies = np.zeros_like(dos)
entropies[:] = -np.inf

for dname in dnames:
    
    print('dir: {}'.format(dname))
    this_p, this_q = get_p_q(dname)
    this_N = this_p*this_q

    assert len(glob.glob('{}/dos_*'.format(dname))) == this_N+1
    this_state = State(np.array([], dtype=int), p=this_p, q=this_q)
    this_max_noo = this_state.n_oo
    this_max_noe = this_state.n_oe

    idx_size = np.where((vals_pq[:,0] == this_p) & (vals_pq[:,1] == this_q))
    assert idx_size[0].size == 1
    idx_size = idx_size[0].item()

    ## WARNING: kc and ko are irritatingly switched
    for this_kc in range(this_N+1):
        this_ko = this_N - this_kc
        idx_ko = np.digitize(this_ko, vals_ko) - 1 

        ds = np.load('{}/dos_p_{:02d}_q_{:02d}_ko_{:03d}.npz'.format(dname, this_p, this_q, this_kc))

        ## dos should sum to this for this p,q,k_o - might be off by 1 since we're casting to an int
        this_omega = int(np.round(binom(this_N, this_ko)))

        # sanity checks
        assert ds['p'].item() == this_p
        assert ds['q'].item() == this_q
        assert ds['ko'].item() == this_kc
        if this_omega != ds['omega_k'].item():
            assert this_omega - ds['omega_k'].item() == 1

        # 'Normalize' the dos, in that its sum must equal this_omega
        this_dos = ds['density']
        assert (this_dos.shape[0] == this_max_noo+1) and (this_dos.shape[1] == this_max_noe+1)

        this_dos /= this_dos.sum()
        this_dos *= this_omega

        assert np.isclose(this_dos.sum(), this_omega)

        this_entropy = np.log(this_dos)

        dos[idx_size, idx_ko, :this_max_noo+1, :this_max_noe+1] = this_dos
        entropies[idx_size, idx_ko, :this_max_noo+1, :this_max_noe+1] = this_entropy

    #assert np.isclose(dos[idx_size].sum(), 2**(this_N))


np.savez_compressed('sam_dos', dos=dos, entropies=entropies, vals_pq=vals_pq, 
                    vals_ko=vals_ko, vals_noo=vals_noo, vals_noe=vals_noe, header='pq, ko, noo, noe')

