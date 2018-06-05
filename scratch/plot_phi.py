from __future__ import division, print_function

import os, glob


fnames = sorted( glob.glob('*/n_v_phi.dat') )
fnames = fnames[1:]

radii = []
peak_sus = []

for i, fname in enumerate(fnames):
    rad = float(os.path.dirname(fname).split('_')[-1]) / 10 # in angstroms
    #rad = labels[i]
    radii.append(rad)

    dat = np.loadtxt(fname)

    n0 = dat[0,1]
    alpha = n0 / np.sqrt(dat[0,2])

    plt.plot(dat[:,0], ((alpha/n0)**1)*dat[:,1], label=r'$r={}$'.format(rad), linewidth=6)
    max_idx = np.argmax(dat[:,2])
    max_phi = dat[max_idx,0]
    peak_sus.append(max_phi)

radii = np.array(radii)
peak_sus = np.array(peak_sus)

plt.ylabel(r'$\langle N_V \rangle_{\phi}$')
#plt.ylabel(r'$\langle (\delta {N_V})^2 \rangle_{\phi}$')
plt.xlabel(r'$\phi$ (kJ/mol)')
plt.legend(loc=1, prop={'size':20})

