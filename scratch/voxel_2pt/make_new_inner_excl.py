from mdtools.fieldwriter import from_cube, RhoField
import numpy as np
import MDAnalysis

from scipy.spatial import cKDTree
import itertools

from scratch.voxel_2pt.gridutils import *

## Exclude voxels too close to the protein
max_dist = 4.0
## exclude voxels if their n_0/0.033 < min_g ('buried' voxels)
min_g = 0.5

ds = np.load('phi_sims/phi_000/init_data.pkl.npz')
n_0 = np.load('phi_sims/ni_weighted.dat.npz')['avg'][:,0]
gridpt_mask = ds['gridpt_mask']
x_bounds = ds['x_bounds']
y_bounds = ds['y_bounds']
z_bounds = ds['z_bounds']

gridpts = construct_gridpts(x_bounds, y_bounds, z_bounds)

## Total number of grid points ##
n_pts_total = gridpts.shape[0]
## Number of points within original distance bounds ##
##  This is the current number of rho_i's
n_pts_red = gridpt_mask.sum()

# Reduced index => global index look up
#  (in other words, the global indices of the reduced gridpts)
red_2_global = np.arange(n_pts_total)[gridpt_mask]

print('N pts reduced: {}'.format(n_pts_red))

gridpts_red = gridpts[gridpt_mask]
save_gridpts('grid_reduced.pdb', gridpts_red, tempfactors=n_0)

## Further reduce points - remove any reduced gridpts further from our new max dist
univ = MDAnalysis.Universe("2pt/prot_shifted.pdb")
prot_h = univ.select_atoms("protein and not name H*")
prot_tree = cKDTree(prot_h.positions)
tree = cKDTree(gridpts_red)

## Grid points that are within max dist of prot
neighbor_list_by_point = prot_tree.query_ball_tree(tree, r=max_dist)
neighbor_list = itertools.chain(*neighbor_list_by_point)
neighbor_idx = np.unique( np.fromiter(neighbor_list, dtype=int) )

# Ids of reduced points that are further from our new distance cutoff
far_pt_idx = np.setdiff1d(np.arange(n_pts_red), neighbor_idx)
# Mask of gridpts reduced that have n_0/0.033 < cutoff
low_n_mask = n_0 < min_g * 0.033

print('Excluding {} pts that are further than {:.2f} A from prot'.format(far_pt_idx.size, max_dist))
print('Excluding {} pts that have n_0 < {:1.2e}'.format(low_n_mask.sum(), min_g*0.033))

# Of *global* indices #
global_mask = gridpt_mask.copy()
global_mask[red_2_global[far_pt_idx]] = False
global_mask[red_2_global[low_n_mask]] = False

# Of *reduced* indices #
red_mask = np.ones(n_pts_red, dtype=bool)
red_mask[far_pt_idx] = False
red_mask[low_n_mask] = False

n_pts_final = global_mask.sum()
print('\nFinal number of points: {} ({} removed from reduced points)'.format(n_pts_final, n_pts_red-n_pts_final))



np.savetxt('red_mask.dat', red_mask, fmt='%1d')

save_gridpts('gridpts_final.pdb', gridpts_red[red_mask], tempfactors=n_0[red_mask])

