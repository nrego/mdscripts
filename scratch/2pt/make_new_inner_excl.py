from mdtools.fieldwriter import from_cube, RhoField
import numpy as np
import MDAnalysis

from scipy.spatial import cKDTree
import itertools

## Exclude voxels too close to the protein
max_dist = 4.0
## exclude voxels if their n_0/0.033 < min_g ('buried' voxels)
min_g = 0.5

ds = np.load('phi_sims/phi_000/init_data.pkl.npz')
gridpt_mask = ds['gridpt_mask']
x_bounds = ds['x_bounds']
y_bounds = ds['y_bounds']
z_bounds = ds['z_bounds']

xx, yy, zz = np.meshgrid(x_bounds[:-1], y_bounds[:-1], z_bounds[:-1], indexing='ij')
gridpts = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T + 0.5
gridpts = gridpts
n_pts_total = gridpts.shape[0]

univ = MDAnalysis.Universe("2pt/prot_shifted.pdb")
prot_h = univ.select_atoms("protein and not name H*")
prot_tree = cKDTree(prot_h.positions)
tree = cKDTree(gridpts)

## Grid points that are further than max dist
neighbor_list_by_point = prot_tree.query_ball_tree(tree, r=max_dist)
neighbor_list = itertools.chain(*neighbor_list_by_point)
neighbor_idx = np.unique( np.fromiter(neighbor_list, dtype=int) )

far_pt_idx = np.setdiff1d(np.arange(n_pts_total), neighbor_idx)

n_0 = np.load("reweight_data/beta_phi_000/ni_reweighted.dat.npz")['rho_water'][0]
buried_mask = n_0/0.033 < min_g

new_mask = np.ones(n_pts_total, dtype=bool)
new_mask[far_pt_idx] = False

mymask = new_mask[gridpt_mask]
mymask[buried_mask] = False

univ = MDAnalysis.Universe("order.pdb")
univ.atoms[mymask].write("excl.pdb")
univ = MDAnalysis.Universe("2pt/out.pdb")
univ.atoms[mymask].write("2pt/exlc.pdb")