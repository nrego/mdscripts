# Analyze results of dynamic_indus (i.e. per-atom, time resolved number of waters and other prot atoms)
#   to locate hydrophobic patches on protein
from __future__ import division, print_function

import MDAnalysis
import os, glob
import numpy as np
import matplotlib 
from matplotlib import cm

base_univ = MDAnalysis.Universe('phi_000/confout.gro')
base_prot_group = base_univ.select_atoms('protein and not name H*')
univ = MDAnalysis.Universe('phi_000/dynamic_volume_water_avg.pdb')
prot_atoms = univ.atoms
assert base_prot_group.n_atoms == prot_atoms.n_atoms

# Note: 0-indexed!
all_atom_lookup = base_prot_group.indices # Lookup table from prot_atoms index to all-H index
atm_indices = prot_atoms.indices

max_water = 33.0 * (4./3.)*np.pi*(0.6)**3

# To determine if atom buried (if less than thresh) or not
avg_water_thresh = 5
# atom is 'dewet' if its average number of is less than this percent of
#   its value at phi=0
per_dewetting_thresh = 0.5
# Threshold for determining if two atoms are correlated
corr_thresh = 0.3

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

for fpath in paths:
    dirname = os.path.dirname(fpath)

    all_h_univ = MDAnalysis.Universe('{}/confout.gro'.format(dirname))
    all_h_univ.atoms.bfactors = 0
    all_prot_atoms = all_h_univ.select_atoms('protein and not name H*')
    univ = MDAnalysis.Universe('{}/dynamic_volume_water_avg.pdb'.format(dirname))
    univ.atoms[~surf_mask].bfactors = max_water
    all_prot_atoms[~surf_mask].bfactors = max_water
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
    all_prot_atoms[surf_mask].bfactors = per_dewet
    univ.atoms.write('{}/per_dewet.pdb'.format(dirname))
    all_h_univ.atoms.write('{}/all_per_dewet.pdb'.format(dirname))

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
    all_h_univ.atoms.bfactors = 1
    
    clusters = []
    dewet_atm_indices = surf_indices[dewet_mask]
    for i, atm_i in enumerate(dewet_atm_indices):
        
        
        corr_i = masked_corr[i]
        corr_atms_i = surf_indices[corr_i > corr_thresh] 
        assert atm_i in corr_atms_i
        
        univ.atoms[corr_atms_i].bfactors = 0
        all_prot_atoms[corr_atms_i].bfactors = 0

        clusters.append(corr_atms_i)

    # Merge into unique clusters
    uniq_clusters = []
    n_clusters = 0
    n_corr_atoms = 0
    for i, cluster in enumerate(clusters):
        if cluster.size == 0:
            continue

        # For each previous cluster j,
        #   check if any shared atoms - if so, merge
        new_group = True
        for j in range(i):
            corr_atms_j = clusters[j]
            if (np.intersect1d(corr_atms_i, cluster)).size > 0:
                new_group = False
                grp_idx = j
                clusters[j] = np.unique(np.concatenate((corr_atms_i, corr_atms_j)))
                break

        if new_group:
            uniq_clusters.append(cluster)
        univ.atoms[cluster].segids = str(i)
        all_prot_atoms[cluster].segids = str(i)

    print("{}: n_clusters: {}  n_corr_atoms: {}".format(dirname, n_clusters, n_corr_atoms))

    univ.atoms.write('{}/correlated_groups.pdb'.format(dirname))
    all_h_univ.atoms.write('{}/all_correlated_groups.pdb'.format(dirname))


