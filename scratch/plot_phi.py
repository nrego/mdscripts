from __future__ import division, print_function

import os, glob


fnames = sorted( glob.glob('sh_*/n_v_phi.dat') )

for fname in fnames[:6]:
    rad = float(os.path.dirname(fname).split('_')[-1]) / 10 # in angstroms
    dat = np.loadtxt(fname)

    plt.plot(dat[:,0], dat[:,1], label=r'$r={}$'.format(rad))

plt.legend()

