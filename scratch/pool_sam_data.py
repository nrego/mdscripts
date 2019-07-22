
import os, glob
from scipy.spatial import cKDTree
from scratch.sam.util import *

def do_fit(pvn, deg):
    x = pvn[:,0]
    y = pvn[:,1]
    z = np.polyfit(x, y, deg)
    p = np.poly1d(z)
    fit = p(x)
    mse = np.mean((fit-y)**2)

    deriv = np.gradient(y, x)
    d = p.deriv()
    fit_deriv = d(x)
    mse_deriv = np.mean((fit_deriv-deriv)**2)


    return (z, mse, mse_deriv)


dirnames = np.sort(glob.glob('../*pattern_sample/*/d_*/trial_0'))
n_dat = dirnames.size + 2 # for k=0 and k=36 

old_ds = np.load('old_sam_pattern_data.dat.npz')
positions = old_ds['positions']

methyl_base = np.zeros(36, dtype=bool)

methyl_pos = np.zeros((n_dat, 36), dtype=bool)
k_vals = np.zeros(n_dat)
energies = np.zeros(n_dat)

beta_phi_stars = np.zeros((n_dat, 36), dtype=np.float32)

poly_4 = np.zeros((n_dat, 5))
poly_5 = np.zeros((n_dat, 6))

pathnames = np.empty(n_dat, dtype=object)

fit_degs = np.arange(2,7)
errs_mse = np.zeros((n_dat, fit_degs.size))
errs_mse_d = np.zeros_like(errs_mse)

for i, dirname in enumerate(dirnames):
    #if dirname == '../pattern_sample/k_27/d_105/trial_0' or dirname == '../pattern_sample/k_29/d_105/trial_0':
    #    continue
    methyl_mask = methyl_base.copy()
    pt_pos = np.loadtxt('{}/this_pt.dat'.format(dirname), dtype=int)
    methyl_mask[pt_pos] = True
    k_ch3 = methyl_mask.sum()

    energy = np.loadtxt('{}/PvN.dat'.format(dirname))[0,1]
    pvn = np.loadtxt('{}/PvN.dat'.format(dirname))
    mask = ~np.ma.masked_invalid(pvn[:,1]).mask
    beta_phi_star = np.loadtxt('{}/beta_phi_i_star.dat'.format(dirname))

    pathnames[i] = dirname
    methyl_pos[i] = methyl_mask
    k_vals[i] = k_ch3
    energies[i] = energy
    beta_phi_stars[i] = beta_phi_star

    for i_deg, deg in enumerate(fit_degs):
        z, mse, mse_d = do_fit(pvn[mask], deg)
        errs_mse[i, i_deg] = mse
        errs_mse_d[i, i_deg] = mse_d
        if deg == 4:
            poly_4[i, ...] = z
        elif deg == 5:
            poly_5[i, ...] = z

# k_00
pathnames[-2] = '../pattern_sample/k_00'
energy = np.loadtxt('../pattern_sample/k_00/PvN.dat')[0,1]
beta_phi_star = np.loadtxt('../pattern_sample/k_00/beta_phi_i_star.dat')
beta_phi_stars[-2] = beta_phi_star
energies[-2] = energy
k_vals[-2] = 0
pvn = np.loadtxt('../pattern_sample/k_00/PvN.dat')
mask = ~np.ma.masked_invalid(pvn[:,1]).mask

for i_deg, deg in enumerate(fit_degs):
    z, mse, mse_d = do_fit(pvn[mask], deg)
    errs_mse[-2, i_deg] = mse
    errs_mse_d[-2, i_deg] = mse_d
    if deg == 4:
        poly_4[-2, ...] = z
    elif deg == 5:
        poly_5[-2, ...] = z

# k_36
pathnames[-1]= '../pattern_sample/k_36'
energy = np.loadtxt('../pattern_sample/k_36/PvN.dat')[0,1]
beta_phi_star = np.loadtxt('../pattern_sample/k_36/beta_phi_i_star.dat')
beta_phi_stars[-1] = beta_phi_star
energies[-1] = energy
k_vals[-1] = 36
methyl_pos[-1][:] = True
pvn = np.loadtxt('../pattern_sample/k_36/PvN.dat')
mask = ~np.ma.masked_invalid(pvn[:,1]).mask

for i_deg, deg in enumerate(fit_degs):
    z, mse, mse_d = do_fit(pvn[mask], deg)
    errs_mse[-1, i_deg] = mse
    errs_mse_d[-1, i_deg] = mse_d
    if deg == 4:
        poly_4[-1, ...] = z
    elif deg == 5:
        poly_5[-1, ...] = z

### TMP ####
indices_to_delete = np.arange(n_dat)[energies == 0]
methyl_pos = np.delete(methyl_pos, indices_to_delete, axis=0)
energies = np.delete(energies, indices_to_delete)
feat_vec = np.delete(methyl_pos, indices_to_delete, axis=0)
poly_4 = np.delete(poly_4, indices_to_delete, axis=0)
beta_phi_stars = np.delete(beta_phi_stars, indices_to_delete, axis=0)
### END TMP ###

np.savez_compressed('sam_pattern_data.dat', pathnames=pathnames, energies=energies, positions=positions, 
                    k_vals=k_vals, methyl_pos=methyl_pos, poly_4=poly_4, poly_5=poly_5, beta_phi_stars=beta_phi_stars)

## Find k_eff_all - enumerate all edge types
pos_ext = gen_pos_grid(12, z_offset=True, shift_y=-3, shift_z=-3)
# patch_idx is list of patch indices in pos_ext 
#   (pos_ext[patch_indices[i]] will give position[i], ith patch point)
d, patch_indices = cKDTree(pos_ext).query(positions, k=1)

nn, nn_ext, dd, dd_ext = construct_neighbor_dist_lists(positions, pos_ext)
edges, ext_indices = enumerate_edges(positions, pos_ext, nn_ext, patch_indices)


edges, ext_indices = enumerate_edges(positions, pos_ext, nn_ext, patch_indices)
n_edges = edges.shape[0]

n_samples = energies.shape[0]

# shape: (n_samples, n_edges, 5)
#   where edge types are:
#   n_mm, n_oo, n_mo, n_me, n_oe
k_eff_all_shape = np.zeros((n_samples, n_edges, 5))

for i, methyl_mask in enumerate(methyl_pos):
    k_eff_all_shape[i, ...] = get_keff_all(methyl_mask, edges, patch_indices)

# n_oo + n_oe
oo = k_eff_all_shape[:,:,1] + k_eff_all_shape[:,:,4]
# n_mo + n_me
mo = k_eff_all_shape[:,:,2] + k_eff_all_shape[:,:,3]
mm = k_eff_all_shape[:,:,0]


# Now only mm, oo, mo;  still have redundancy
k_eff_all_shape = np.dstack((mm, oo, mo))

np.save('k_eff_all.dat', k_eff_all_shape)