# Analyze results of dynamic_indus (i.e. per-atom, time resolved number of waters and other prot atoms)
#   to locate hydrophobic patches on protein
from __future__ import division, print_function

import MDAnalysis
import os, glob
import numpy as np
import matplotlib 
from matplotlib import cm

fpaths = sorted( glob.glob('phi_*/roc.dat') )

for path in fpaths:
    dirname = os.path.dirname(path)
    phi_val = float(dirname.split('_')[-1]) / 10.0

    roc = np.loadtxt(path)

    plt.plot(roc[:,1], roc[:,2], '-o', label=r'$\phi={}$'.format(phi_val), linewidth=2, markersize=6)

plt.plot(0,1,'ok')
plt.plot([0,1],[0,1], 'k--')

plt.legend()

plt.xlabel('FPR')
plt.ylabel('TPR')
plt.tight_layout()

plt.show()