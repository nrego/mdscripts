import glob, os, sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

from mdtools import dr

from constants import k

import MDAnalysis

import argparse

mpl.rcParams.update({'axes.labelsize': 50})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 40})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':10})

from IPython import embed

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['nofilter', 'phob', 'dewet'], default='nofilter')
args = parser.parse_args()

hydropathy = np.loadtxt('../bound/hydropathy_mask.dat').astype(bool)
buried_mask = np.loadtxt('../bound/buried_mask.dat').astype(bool)
surf_mask = ~buried_mask 

if args.mode == 'nofilter':
    actual_contacts = np.loadtxt('../bound/actual_contact_mask.dat').astype(bool)
    post = ''
elif args.mode == 'phob':
    actual_contacts = np.loadtxt('../bound/actual_contact_mask_phob.dat').astype(bool) #& hydropathy
    post = '_phob'
    surf_mask = surf_mask & hydropathy
elif args.mode == 'dewet':
    actual_contacts = np.loadtxt('../bound/actual_contact_mask_dewet.dat').astype(bool) #& hydropathy
    post = '_dewet'

print('contacts: {}'.format(actual_contacts.sum()))

fnames = sorted(glob.glob('phi_*'))

phi_vals = []
tpr = []
fpr = []
dist = []
sus = []

f1s = []

all_info = []

for dirname in fnames:
    ds = dr.loadPhi('../prod/phi_sims/{}/phiout.dat'.format(dirname))
    var = ds.data[500:]['$\~N$'].var()
    sus.append(var)
    phi_val = float(dirname.split('_')[-1])/10
    phi_vals.append(phi_val)

    struct = MDAnalysis.Universe('{}/pred_contact.pdb'.format(dirname))
    struct.atoms.tempfactors = -2
    surf_atoms = struct.atoms[surf_mask]

    pred_contact_mask = np.loadtxt('{}/pred_contact_mask.dat'.format(dirname)).astype(bool)

    tp_mask = pred_contact_mask[surf_mask] & actual_contacts[surf_mask]
    fp_mask = pred_contact_mask[surf_mask] & ~actual_contacts[surf_mask]
    tn_mask = ~pred_contact_mask[surf_mask] & ~actual_contacts[surf_mask]
    fn_mask = ~pred_contact_mask[surf_mask] & actual_contacts[surf_mask]

    surf_atoms[tp_mask].tempfactors = 1
    surf_atoms[fp_mask].tempfactors = 0
    surf_atoms[fn_mask].tempfactors = -1

    struct.atoms.write('{}/pred_contact_tp_fp{}.pdb'.format(dirname, post))

    tp = tp_mask.sum()
    fp = fp_mask.sum()
    tn = tn_mask.sum()
    fn = fn_mask.sum()

    all_info.append([tp,fp,tn,fn])

    #if phi_val == 5.6:
    #    embed()

    try:
        this_prec = float(tp)/(tp+fp)
    except:
        this_prec = 0

    this_tpr = float(tp)/(tp+fn)
    this_fpr = float(fp)/(fp+tn)

    try:
        this_f1 = 2/((1/this_prec) + (1/this_tpr))
    except ZeroDivisionError:
        this_f1 = 0
    f1s.append(this_f1)

    tpr = np.append(tpr, this_tpr)
    fpr = np.append(fpr, this_fpr)

    this_dist = np.sqrt((this_tpr-1)**2 + this_fpr**2)
    dist = np.append(dist, this_dist)

min_idx = np.argmin(dist)
max_idx = np.argmax(sus)

phi_vals = np.array(phi_vals)
beta = 1/(300 * k)
all_info = np.array(all_info)

np.savetxt('corr_phi.dat', np.array([phi_vals[min_idx], phi_vals[max_idx]]))
np.savetxt('performance.dat', np.hstack((phi_vals[:,None],all_info)), header='phi  tp   fp   tn   fn')
np.savetxt('dewet_dist_with_phi{}.dat'.format(post), np.vstack((phi_vals, sus, dist)).T)
np.savetxt('dewet_roc{}.dat'.format(post), np.vstack((phi_vals, fpr, tpr)).T)
plt.plot(fpr, tpr, 'o')
plt.show()

plt.plot(phi_vals, dist, '-o')
plt.show()

plt.plot(phi_vals, f1s, '-o')
plt.show()



