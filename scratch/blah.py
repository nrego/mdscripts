# Analyze results of dynamic_indus (i.e. per-atom, time resolved number of waters and other prot atoms)
#   to locate hydrophobic patches on protein
from __future__ import division, print_function

import MDAnalysis
import os, glob
import numpy as np
import matplotlib 
from matplotlib import cm


# Plot all ROC curves for each phi val

fnames = sorted(glob.glob('phi*/roc.dat'))

plot_phis = np.array([ 1, 3, 4, 4.5, 5, 5.5, 6, 8, 10, 30]).astype(float)

all_phis = []
all_min_dists = []

plt.plot(0,1,'ko')
plt.plot([0,1],[0,1], 'k--')
plt.axvline(0, linestyle='--', color='k', alpha=0.5)
plt.plot([-0.05,1.05], [1,1], 'k--', alpha=0.5)
plt.xlabel('FPR')
plt.ylabel('TPR')

for fname in fnames:
    dirname = os.path.dirname(fname)
    phi_val = int(dirname.split('_')[-1]) / 10.0
    all_phis.append(phi_val)

    roc_dat = np.loadtxt(fname)

    dists = np.sqrt( (roc_dat[:,1])**2 + (roc_dat[:,2] - 1)**2 )
    arg_min_dist = np.argmin(dists)
    min_dist = dists[arg_min_dist]
    all_min_dists.append(min_dist)
    best_thresh = roc_dat[arg_min_dist, 0]

    print("phi: {}".format(phi_val))
    print("   best threshold: {}".format(best_thresh))
    print("   dist: {}".format(min_dist))

    if phi_val in plot_phis:
        plt.plot(roc_dat[:,1], roc_dat[:,2], '-o', label=r'$\phi={}$'.format(phi_val))

plt.ylim(-0.05, 1.05)
plt.xlim(-0.05, 1.05)
plt.legend()
plt.tight_layout()
plt.show()

plt.plot(all_phis, all_min_dists, '-o')
plt.xlabel(r'$\phi$ (kJ/mol)')
plt.xlim(0,10)
plt.ylabel('min_dist')
plt.tight_layout()
