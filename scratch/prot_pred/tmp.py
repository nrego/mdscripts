import glob, os, sys
import numpy as np
from matplotlib import pyplot as plt

actual_contacts = np.loadtxt('../bound/actual_contact_mask_phob.dat').astype(bool)
buried_mask = np.loadtxt('../bound/buried_mask.dat').astype(bool)
fnames = sorted(glob.glob('phi_*'))

phi_vals = []
tp = []
tn = []
fp = []
fn = []
for dirname in fnames:
    phi_val = float(dirname.split('_')[-1])/10
    phi_vals.append(phi_val)

    pred_contact = np.loadtxt('{}/pred_contact_mask.dat'.format(dirname)).astype(bool)

    tp_mask = pred_contact[~buried_mask] & actual_contacts[~buried_mask]
    tn_mask = ~pred_contact[~buried_mask] & ~actual_contacts[~buried_mask]

    fp_mask = pred_contact[~buried_mask] & ~actual_contacts[~buried_mask]
    fn_mask = ~pred_contact[~buried_mask] & actual_contacts[~buried_mask]

    tp.append(tp_mask.sum())
    tn.append(tn_mask.sum())
    fp.append(fp_mask.sum())
    fn.append(fn_mask.sum())

tp = np.array(tp).astype(float)
tn = np.array(tn).astype(float)
fp = np.array(fp).astype(float)
fn = np.array(fn).astype(float)

tpr = tp/(tp+fn)
fpr = fp/(fp+tn)

dist = np.sqrt((tpr - 1)**2 + fpr**2)
plt.plot(fpr, tpr, 'o-')

plt.show()

plt.plot(phi_vals, dist)

plt.show()