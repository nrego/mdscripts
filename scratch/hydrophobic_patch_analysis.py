# Analyze results of dynamic_indus (i.e. per-atom, time resolved number of waters and other prot atoms)
#   to locate hydrophobic patches on protein
from __future__ import division, print_function

import MDAnalysis
import os, glob
import numpy as np
import matplotlib 
from matplotlib import cm

univ = MDAnalysis.Universe('phi_000/dynamic_volume_water_avg.pdb')
prot_atoms = univ.atoms
# Note: 0-indexed!
atm_indices = prot_atoms.indices

max_water = 33.0 * (4./3.)*np.pi*(0.6)**3

# To determine if atom buried (if less than thresh) or not
avg_water_thresh = 5
# atom is 'dewet' if its average number of is less than this percent of
#   its value at phi=0
per_dewetting_thresh = 0.5
# Threshold for determining if two atoms are correlated
corr_thresh = 0.2

ds_0 = np.load('phi_000/rho_data_dump.dat.npz')

# shape: (n_atoms, n_frames)
rho_solute0 = ds_0['rho_solute'].T
rho_water0 = ds_0['rho_water'].T

assert rho_solute0.shape[0] == rho_water0.shape[0] == prot_atoms.n_atoms
n_frames = rho_water0.shape[1]

# averages
rho_avg_solute0 = rho_solute0.mean(axis=1)
rho_avg_water0 = rho_water0.mean(axis=1)

# Find surface atoms...
surf_mask = rho_avg_water0 > avg_water_thresh
np.savetxt('surf_mask.dat', surf_mask, fmt='%d')
rho_solute0 = rho_solute0[surf_mask]
rho_water0 = rho_water0[surf_mask]
rho_avg_solute0 = rho_avg_solute0[surf_mask]
rho_avg_water0 = rho_avg_water0[surf_mask]

surf_indices = atm_indices[surf_mask]

# variance and covariance
delta_rho_water0 = rho_water0 - rho_avg_water0[:,np.newaxis]
cov_rho_water0 = np.dot(delta_rho_water0, delta_rho_water0.T) / n_frames
rho_var_water0 = cov_rho_water0.diagonal()

del ds_0

paths = sorted(glob.glob('phi*/rho_data_dump.dat.npz'))

for fpath in [paths[5]]:
    dirname = os.path.dirname(fpath)

    univ = MDAnalysis.Universe('{}/dynamic_volume_water_avg.pdb'.format(dirname))
    univ.atoms[~surf_mask].bfactors = max_water
    univ.atoms.write('{}/surf_masked.pdb'.format(dirname))

    ds = np.load(fpath)
    rho_solute = (ds['rho_solute'].T)[surf_mask]
    rho_water = (ds['rho_water'].T)[surf_mask]
    del ds

    assert rho_water.shape[0] == surf_mask.sum()
    # can't assume all datasets have the same number of frames in analysis
    #    (though they should!)
    n_frames = rho_water.shape[1]

    rho_avg_solute = rho_solute.mean(axis=1)
    rho_avg_water = rho_water.mean(axis=1)

    # percentage each atom is dewet (w.r.t. its phi=0.0 value)
    per_dewet = (rho_avg_water/rho_avg_water0)
    univ.atoms[surf_mask].bfactors = per_dewet
    univ.atoms.write('{}/per_dewet.pdb'.format(dirname))

    dewet_mask = per_dewet < per_dewetting_thresh
    dewet_mask_2d = np.matmul(dewet_mask[:,np.newaxis], dewet_mask[np.newaxis,:])

    delta_rho_water = rho_water - rho_avg_water[:,np.newaxis]
    cov_rho_water = np.dot(delta_rho_water, delta_rho_water.T) / n_frames
    rho_var_water = cov_rho_water.diagonal()

    cov_norm = np.sqrt( np.matmul(rho_var_water[:,np.newaxis], rho_var_water[np.newaxis,:]) )

    # Normalized covariance matrix (correlation coefs)
    corr_rho_water = cov_rho_water / cov_norm

    masked_corr = corr_rho_water[dewet_mask,:]
    #masked_corr = masked_corr[:,dewet_mask]

    corr_lim = np.abs(corr_rho_water).max()

    norm = matplotlib.colors.Normalize(-1,1)

    im = plt.imshow(corr_rho_water, interpolation='nearest', cmap=cm.bwr, norm=norm, aspect='auto')
    cb = plt.colorbar(im)
    plt.title(dirname)
    #plt.show()

    ## Put together list of atoms with correlated water numbers
    univ.atoms.bfactors = 1
    n_corr_atoms = 0
    marked_idxs = []
    for idx, atm_i in enumerate(surf_indices[dewet_mask]):
        
        this_indices = []
        corr_i = masked_corr[idx]
        cand_atms_i = surf_indices[corr_i > corr_thresh] 
        n_corr_atoms += cand_atms_i.size
        univ.atoms[cand_atms_i].bfactors = 0

        seen = False
        seen_idx = 0
        for cand_idx in cand_atms_i:
            this_indices.append(cand_idx)

            for j in range(idx):
                if cand_idx in marked_idxs[j]:
                    seen = True
                    seen_idx = j
                    break
            if seen:
                break
        if seen:
            univ.atoms[cand_atms_i].segids = str(seen_idx)
        else:
            univ.atoms[cand_atms_i].segids = str(idx)


        marked_idxs.append(cand_atms_i)

    print("{}: n_corr_atoms: {}".format(dirname, n_corr_atoms))

    univ.atoms.write('{}/correlated_groups.pdb'.format(dirname))