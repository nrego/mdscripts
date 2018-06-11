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


from constants import k

fig = plt.figure(figsize=(8.5,7))
beta = 1/(k*300)
dat = np.loadtxt('n_v_phi.dat')
err_dat = np.loadtxt('err_n_v_phi.dat')

plt.plot(beta*dat[:,0], dat[:,1], 'k-', linewidth=6)

plt.xlabel(r'$\beta \phi$')
plt.ylabel(r'$\langle N_V \rangle_{\phi}$')

plt.xlim(0, beta*10)


plt.tight_layout()
plt.savefig('/Users/nickrego/Desktop/ubiq_n_v_phi.pdf')
plt.show()



fig = plt.figure(figsize=(9,7))


plt.errorbar(beta*dat[:,0], dat[:,2], yerr=err_dat[:,1], fmt='k-', linewidth=6, elinewidth=3)

plt.xlabel(r'$\beta \phi$')
plt.ylabel(r'$- \frac{d\langle N_V \rangle_{\phi}}{d \beta \phi}$')

plt.xlim(0, beta*10)


plt.tight_layout()
plt.savefig('/Users/nickrego/Desktop/ubiq_sus_v_phi.pdf')
plt.show()