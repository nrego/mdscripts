import glob, os, sys

actual_contacts = np.loadtxt('../bound/actual_contact_mask.dat').astype(bool)
fnames = sorted(glob.glob('phi_*'))

phi_vals = []
tpr = []
fpr = []
for dirname in fnames:
    phi_val = float(dirname.split('_')[-1])/10
    phi_vals.append(phi_val)
