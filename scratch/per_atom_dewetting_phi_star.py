# Analyze results of dynamic_indus (i.e. per-atom, time resolved number of waters and other prot atoms)
#   to locate hydrophobic patches on protein
from __future__ import division, print_function

import MDAnalysis
import os, glob
import numpy as np
import matplotlib 
from matplotlib import cm

fnames = sorted(glob.glob('phi_*/rho_data_dump.dat.npz'))
for fpath in fnames:
    dirname = os.path.dirname(fpath)
    phi_val = float(dirname.split('_')[-1]) / 10

    print("PHI: {}".format(phi_val))
    # Only look at given phi - compare diffrent threshold values
    #fpath = 'phi_055/rho_data_dump.dat.npz'

    base_univ = MDAnalysis.Universe('phi_000/confout.gro')
    base_prot_group = base_univ.select_atoms('protein and not name H*')
    univ = MDAnalysis.Universe('phi_000/dynamic_volume_water_avg.pdb')
    prot_atoms = univ.atoms
    assert base_prot_group.n_atoms == prot_atoms.n_atoms


    surf_mask = np.loadtxt('surf_mask.dat').astype(bool)
    buried_mask = ~surf_mask
    print("n_tot: {}  n_surf: {}  n_buried: {}".format(univ.atoms.n_atoms, surf_mask.sum(), buried_mask.sum()))

    try:
        nat_contacts = MDAnalysis.Universe('contacts.pdb').atoms[surf_mask].bfactors == 0
        print("total native contacts: {}".format(nat_contacts.sum()))
    except:
        nat_contacts = None
    # weight for calculating per-atom norm density
    wt = 0.5 

    # Note: 0-indexed!
    all_atom_lookup = base_prot_group.indices # Lookup table from prot_atoms index to all-H index
    atm_indices = prot_atoms.indices

    max_water = 33.0 * (4./3.)*np.pi*(0.6)**3

    # To determine if atom buried (if avg waters fewer than thresh) or not
    avg_water_thresh = 5


    ds_0 = np.load('phi_000/rho_data_dump.dat.npz')

    # shape: (n_atoms, n_frames)
    rho_solute0 = ds_0['rho_solute'].T
    rho_water0 = ds_0['rho_water'].T

    assert rho_solute0.shape[0] == rho_water0.shape[0] == prot_atoms.n_atoms
    n_frames = rho_water0.shape[1]

    # averages per heavy
    rho_avg_solute0 = rho_solute0.mean(axis=1)
    rho_avg_water0 = rho_water0.mean(axis=1)

    max_solute = rho_avg_solute0.max()
    max_water = rho_avg_water0.max()

    # variance and covariance
    delta_rho_water0 = rho_water0 - rho_avg_water0[:,np.newaxis]
    cov_rho_water0 = np.dot(delta_rho_water0, delta_rho_water0.T) / n_frames
    rho_var_water0 = cov_rho_water0.diagonal()

    del ds_0

    #paths = sorted(glob.glob('phi*/rho_data_dump.dat.npz'))

    all_h_univ = MDAnalysis.Universe('{}/confout.gro'.format(dirname))
    all_h_univ.atoms.bfactors = 0
    all_prot_atoms = all_h_univ.select_atoms('protein and not name H*')
    univ = MDAnalysis.Universe('{}/dynamic_volume_water_avg.pdb'.format(dirname))

    ds = np.load(fpath)
    rho_solute = (ds['rho_solute'].T)
    rho_water = (ds['rho_water'].T)
    del ds

    # can't assume all datasets have the same number of frames in analysis
    #    (though they should!)
    n_frames = rho_water.shape[1]

    rho_avg_solute = rho_solute.mean(axis=1)
    rho_avg_water = rho_water.mean(axis=1)

    # New, normalized water and protein density
    #   hopefully accounts for case when exposed atom becomes buried at phi>0
    #   (and would possibly be a false dewet positive otherwise)
    per_dewet = wt*(rho_avg_water/rho_avg_water0) + (1-wt)*(rho_avg_solute/rho_avg_solute0)
    per_dewet_just_prot = rho_avg_water/rho_avg_water0
    rho_norm = (rho_avg_water/max_water) + (rho_avg_solute/max_solute)

    univ.atoms.bfactors = per_dewet
    univ.atoms[buried_mask].bfactors = 1
    all_prot_atoms.bfactors = per_dewet
    all_prot_atoms[buried_mask].bfactors = 1
    all_h_univ.atoms.write('{}/all_per_atom_norm.pdb'.format(dirname))
    univ.atoms.write('{}/per_atom_norm.pdb'.format(dirname))

    univ.atoms.bfactors = per_dewet_just_prot
    univ.atoms[buried_mask].bfactors = 1
    all_prot_atoms.bfactors = per_dewet_just_prot
    all_prot_atoms[buried_mask].bfactors = 1
    all_h_univ.atoms.write('{}/all_per_atom_norm.pdb'.format(dirname))
    univ.atoms.write('{}/per_atom_norm.pdb'.format(dirname))

    univ.atoms.bfactors = rho_norm
    univ.atoms[buried_mask].bfactors = 1
    all_prot_atoms.bfactors = rho_norm
    all_prot_atoms[buried_mask].bfactors = 1
    all_h_univ.atoms.write('{}/all_global_norm.pdb'.format(dirname))
    univ.atoms.write('{}/global_norm.pdb'.format(dirname))

    thresholds = np.arange(0, 1.05, 0.05)
    roc = np.zeros((thresholds.size, 2))

    for i, per_dewetting_thresh in enumerate(thresholds):
        print('phi={}'.format(phi_val))
        dewet_mask = (per_dewet_just_prot[surf_mask] < per_dewetting_thresh)

        print("S: {}  n_dewet: {} of {}".format(per_dewetting_thresh, dewet_mask.sum(), surf_mask.sum()))

        if nat_contacts is not None:
            true_pos = dewet_mask & nat_contacts
            false_pos = dewet_mask & ~nat_contacts

            true_neg = ~dewet_mask & ~nat_contacts
            false_neg = ~dewet_mask & nat_contacts

            assert dewet_mask.sum() == true_pos.sum() + false_pos.sum()

            print("  total predicted: {}  true pos: {}  false pos: {}  true neg: {}  false_neg: {}".format(dewet_mask.sum(), true_pos.sum(), false_pos.sum(), true_neg.sum(), false_neg.sum()))

            tpr = true_pos.sum() / ( true_pos.sum()+false_neg.sum() )
            fpr = false_pos.sum() / ( true_neg.sum()+false_pos.sum() )
            acc = (true_pos.sum() + true_neg.sum())/univ.atoms.n_atoms
            f1 = (2*true_pos.sum())/(2*true_pos.sum()+false_pos.sum()+false_neg.sum())
            print("  TPR: {}  FPR: {}".format(tpr,fpr))
            print("  ACC: {}".format(acc))
            print("  F1: {}".format(f1))
            roc[i] = fpr, tpr


    plt.plot(roc[:,0], roc[:,1], '-o')
    plt.plot(0,1,'o')
    plt.plot([0,1],[0,1], '--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(r'$\phi={}$'.format(phi_val))
    plt.tight_layout()
    #plt.show()
    roc_arr = np.hstack((thresholds[:,None], roc)).squeeze()
    np.savetxt('{}/roc_meth.dat'.format(dirname), roc_arr)


