from __future__ import division, print_function

import os, glob


fnames = sorted( glob.glob('sh_*/n_v_phi.dat') )

radii = []
peak_sus = []
for fname in fnames[:2]:
    rad = float(os.path.dirname(fname).split('_')[-1]) / 10 # in angstroms
    radii.append(rad)

    dat = np.loadtxt(fname)

    plt.plot(dat[:,0], dat[:,2], label=r'$r={}$'.format(rad))
    max_idx = np.argmax(dat[:,2])
    max_phi = dat[max_idx,0]
    peak_sus.append(max_phi)

plt.legend()

