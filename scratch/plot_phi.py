from __future__ import division, print_function

import os, glob


fnames = sorted( glob.glob('*/n_v_phi.dat') )

radii = []
peak_sus = []
labels=(r'$\rm{CH}_3$', r'$\rm{OH}$')

for i, fname in enumerate(fnames):
    rad = float(os.path.dirname(fname).split('_')[-1]) / 10 # in angstroms
    #rad = labels[i]
    radii.append(rad)

    dat = np.loadtxt(fname)

    plt.plot(dat[:,0], dat[:,2], label=r'$r={}$'.format(rad), linewidth=6)
    max_idx = np.argmax(dat[:,2])
    max_phi = dat[max_idx,0]
    peak_sus.append(max_phi)

#plt.ylabel(r'$\langle N_V \rangle_{\phi}$')
plt.ylabel(r'$\langle (\delta {N_V})^2 \rangle_{\phi}$')
plt.xlabel(r'$\phi$ (kJ/mol)')
plt.legend(loc=1, prop={'size':20})

