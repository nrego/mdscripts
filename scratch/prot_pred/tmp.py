import glob, os, sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

from mdtools import dr

mpl.rcParams.update({'axes.labelsize': 50})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 40})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':10})

actual_contacts = np.loadtxt('../bound/actual_contact_mask.dat').astype(bool)
#actual_contacts = np.loadtxt('../bound/actual_contact_mask_phob.dat').astype(bool)
#actual_contacts = np.loadtxt('../bound/actual_contact_mask_dewet.dat').astype(bool)
print('contacts: {}'.format(actual_contacts.sum()))
buried_mask = np.loadtxt('../bound/buried_mask.dat').astype(bool)
surf_mask = ~buried_mask
fnames = sorted(glob.glob('phi_*'))

phi_vals = []
tpr = []
fpr = []
dist = []
sus = []

for dirname in fnames:
    ds = dr.loadPhi('../prod/phi_sims/{}/phiout.dat'.format(dirname))
    var = ds.data[500:]['$\~N$'].var()
    sus.append(var)
    phi_val = float(dirname.split('_')[-1])/10
    phi_vals.append(phi_val)

    pred_contact_mask = np.loadtxt('{}/pred_contact_mask.dat'.format(dirname)).astype(bool)
    tp = np.sum(pred_contact_mask[surf_mask] & actual_contacts[surf_mask])
    fp = np.sum(pred_contact_mask[surf_mask] & ~actual_contacts[surf_mask])
    tn = np.sum(~pred_contact_mask[surf_mask] & ~actual_contacts[surf_mask])
    fn = np.sum(~pred_contact_mask[surf_mask] & actual_contacts[surf_mask])

    this_tpr = float(tp)/(tp+fn)
    this_fpr = float(fp)/(fp+tn)

    tpr = np.append(tpr, this_tpr)
    fpr = np.append(fpr, this_fpr)

    this_dist = np.sqrt((this_tpr-1)**2 + this_fpr**2)
    dist = np.append(dist, this_dist)

min_idx = np.argmin(dist)
max_idx = np.argmax(sus)

np.savetxt('corr_phi.dat', np.array([phi_vals[min_idx], phi_vals[max_idx]]))
#np.savetxt()
plt.plot(fpr, tpr, '-o')
plt.show()

plt.plot(phi_vals, dist, '-o')
plt.show()