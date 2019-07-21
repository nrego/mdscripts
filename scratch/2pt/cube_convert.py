from mdtools.fieldwriter import from_cube, RhoField
import numpy as np
import MDAnalysis

from scipy.spatial import cKDTree
import itertools

au = 0.5292
# avg waters / nm^3
rho, rho_shape, gridpts_old = from_cube("grid-54x54x54_pot3D-numDens.cube")
dg, dg_shape, _ = from_cube("grid-54x54x54_3D-2PT-dG_shifted.cube")
rho_new, rho_shape, gridpts = from_cube("grid-54x54x54_pot3D-numDens_shifted.cube")
assert np.array_equal(rho, rho_new)
# dg is kJ/(mol * water)
n_pts_total = gridpts.shape[0]

ds = np.load("../phi_sims/phi_000/init_data.pkl.npz")
x_bounds = ds['x_bounds']
y_bounds = ds['y_bounds']
z_bounds = ds['z_bounds']

xx, yy, zz = np.meshgrid(x_bounds[:-1], y_bounds[:-1], z_bounds[:-1], indexing='ij')
grid2 = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T
grid2 += 0.5

assert np.array_equal(gridpts, grid2)


gridpt_mask = ds['gridpt_mask']

new_mask = rho>48

mask = gridpt_mask

incl_voxels = gridpts[mask]
n_inc_pts = mask.sum()

univ = MDAnalysis.Universe.empty(n_inc_pts, n_inc_pts, atom_resindex=np.arange(n_inc_pts), trajectory=True)
univ.add_TopologyAttr('name')
univ.add_TopologyAttr('resname')
univ.add_TopologyAttr('id')
univ.add_TopologyAttr('tempfactors')

univ.residues.resnames = 'V'
univ.atoms.names = 'V'
univ.atoms.tempfactors = dg[mask] 
univ.atoms.positions = gridpts[mask] 

np.savetxt('dg_voxel.dat', dg[mask])
np.savetxt('dg_vol_voxel.dat', dg[mask] * rho[mask])

univ.atoms.write('out.pdb')  ## For testing we are converting correctly...

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
