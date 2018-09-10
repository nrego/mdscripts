from __future__ import division, print_function
  

import MDAnalysis
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({'axes.labelsize': 50})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 40})
mpl.rcParams.update({'axes.titlesize': 50})
mpl.rcParams.update({'legend.fontsize':30})

def plt_errorbars(bb, vals, errs, **kwargs):
    plt.fill_between(bb, vals-errs, vals+errs, alpha=0.5, **kwargs)

from constants import k

fig = plt.figure(figsize=(8.5,7))
beta = 1#/(k*300)
dat = np.loadtxt('n_v_phi.dat')
err_dat = dat[:,2]

plt.plot(beta*dat[:,0], dat[:,1], 'k-', linewidth=6)

plt.xlabel(r'$\beta \phi$')
plt.ylabel(r'$\langle N_v \rangle_{\phi}$')

plt.xlim(0, 4)


plt.tight_layout()
plt.savefig('/Users/nickrego/Desktop/ubiq_n_v_phi.pdf')
plt.show()



fig = plt.figure(figsize=(9,7))

dat = np.loadtxt('var_n_v_phi.dat')
#plt.errorbar(beta*dat[:,0], dat[:,1], yerr=dat[:,2], fmt='k-', linewidth=6, elinewidth=3)
plt.plot(dat[:,0], dat[:,1], 'k-', linewidth=6)
plt_errorbars(dat[:,0], dat[:,1], dat[:,2], color='k')

plt.xlabel(r'$\beta \phi$')
plt.ylabel(r'$\chi_v$')

plt.xlim(0, 4)


plt.tight_layout()
plt.savefig('/Users/nickrego/Desktop/ubiq_sus_v_phi.pdf')
plt.show()


