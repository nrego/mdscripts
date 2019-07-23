from mdtools.fieldwriter import from_cube, RhoField
import numpy as np
import MDAnalysis

from scipy.spatial import cKDTree
import itertools

from scratch.voxel_2pt.gridutils import *

au = 0.5292
# avg waters / nm^3
rho, rho_shape, gridpts_old = from_cube("grid-54x54x54_pot3D-numDens.cube")
dg, dg_shape, _ = from_cube("grid-54x54x54_3D-2PT-dG_shifted.cube")
rho_new, rho_shape, gridpts = from_cube("grid-54x54x54_pot3D-numDens_shifted.cube")
assert np.array_equal(rho, rho_new)


## Print out Shift Info (to align this protein to our system) ##
###########################

u = MDAnalysis.Universe("prot.pdb")
u_ref = MDAnalysis.Universe("../phi_sims/equil.gro")

p = u.select_atoms('protein')
p_ref = u_ref.select_atoms('protein')

shift = p_ref.positions - p.positions

# Shift vector, in angstroms
shift = np.round(shift[0], 4)
# Voxel origin, original (in angstroms)
old_origin = gridpts_old[0]

print("Shift vector (A): {}".format(shift))
print("old origin (A): {}".format(old_origin))
new_origin = np.round(old_origin + shift, 4)
print("new origin (A): {}".format(new_origin))

print("new origin (atomic units): {}".format(np.round(new_origin/au, 6)))

##########################


## Make sure our shifted voxels line up correctly ##

n_pts_total = gridpts.shape[0]

ds = np.load("../phi_sims/phi_000/init_data.pkl.npz")
x_bounds = ds['x_bounds']
y_bounds = ds['y_bounds']
z_bounds = ds['z_bounds']

grid2 = construct_gridpts(x_bounds, y_bounds, z_bounds)
assert np.allclose(gridpts, grid2)


## Output info from dx sets (dg, num density) according to supplied voxel mask
gridpt_mask = ds['gridpt_mask']

gridpts_red = gridpts[gridpt_mask]
n_pts_red = gridpt_mask.sum()

np.savetxt('dg_voxel.dat', dg[gridpt_mask])
np.savetxt('dg_vol_voxel.dat', dg[gridpt_mask] * rho[gridpt_mask])
np.savetxt('rho_voxel.dat', rho[gridpt_mask])


save_gridpts('num_density.pdb', gridpts_red, tempfactors=rho[gridpt_mask])

