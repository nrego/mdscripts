import matplotlib as mpl

import os, glob

fnames = glob.glob('*/bound/hydropathy_mask.dat')

for fname in fnames:

    subdir = os.path.dirname(fname)

    contact_mask = np.loadtxt('{}/actual_contact_mask.dat'.format(subdir)).astype(bool)
    buried_mask = np.loadtxt('{}/buried_mask.dat'.format(subdir)).astype(bool)
    hydropathy_mask = np.loadtxt('{}/hydropathy_mask.dat'.format(subdir)).astype(bool)

    hydrophobic_contacts = hydropathy_mask & contact_mask

    print('sys: {}'.format(subdir))
    print('  n_contacts: {}'.format(contact_mask.sum()))
    print('  hydrophobic frac: {:0.2f}'.format(float(hydrophobic_contacts.sum())/contact_mask.sum()))

#######################
import matplotlib as mpl
from matplotlib import rc 
import os, glob

homedir = os.environ['HOME']

mpl.rcParams.update({'axes.labelsize': 20})
mpl.rcParams.update({'xtick.labelsize': 15})
mpl.rcParams.update({'ytick.labelsize': 15})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':10})

from constants import k

fnames = glob.glob('*/pred/performance.dat')
beta = 1/(k * 300)

sys_names = []
perf = np.array([])
tprs = np.array([])
precs = np.array([])
phis = np.array([])

for fname in fnames:
    

    dat = np.loadtxt(fname)

    tp = dat[:,1]
    fp = dat[:,2]
    tn = dat[:,3]
    fn = dat[:,4]

    tpr = tp/(tp+fn)
    fpr = fp/(tn+fp)

    prec = tp/(tp+fp)


    f1 = 2/((1/tpr) + (1/prec))
    best_perf = np.nanargmax(f1)

    sys_names.append(os.path.dirname(fname))
    perf = np.append(perf, f1[best_perf])
    tprs = np.append(tprs, tpr[best_perf])
    precs = np.append(precs, prec[best_perf])
    phis = np.append(phis, beta*dat[best_perf,0])


fig, ax = plt.subplots(figsize=(8,5))

#ax.bar(np.arange(len(fnames)), perf, color='k', width=0.8)
ax.bar(np.arange(len(fnames)), phis, color='k', width=0.8)
