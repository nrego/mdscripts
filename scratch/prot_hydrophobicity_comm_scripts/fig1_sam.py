from __future__ import division, print_function
  

import MDAnalysis
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({'axes.labelsize': 50})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 40})
mpl.rcParams.update({'axes.titlesize': 50})
mpl.rcParams.update({'legend.fontsize':25})


from constants import k

# N v phi
fig = plt.figure(figsize=(8.5,7))
beta = 1/(k*300)
ch3_dat = np.loadtxt('CH3_Disks/n_v_phi.dat')
oh_dat = np.loadtxt('OH_Disks/n_v_phi.dat')

plt.plot(beta*ch3_dat[:,0], ch3_dat[:,1], 'r-', label=r'$\rm{CH}_3$', linewidth=6)
plt.plot(beta*oh_dat[:,0], oh_dat[:,1], 'b-', label=r'$\rm{OH}$', linewidth=6)

plt.xlabel(r'$\beta \phi$')
plt.ylabel(r'$\langle N_V \rangle_{\phi}$')

plt.xlim(0,beta*5)
plt.xticks([0,1,2])
plt.ylim(0,130)

plt.legend(loc=1)
plt.tight_layout()
plt.savefig('/Users/nickrego/Desktop/sam_n_v_phi.pdf')
plt.show()

# dN/dphi v phi
fig = plt.figure(figsize=(9,7))
beta = 1/(k*300)
ch3_dat = np.loadtxt('CH3_Disks/n_v_phi.dat')
oh_dat = np.loadtxt('OH_Disks/n_v_phi.dat')

plt.plot(beta*ch3_dat[:,0], ch3_dat[:,2], 'r-', label=r'$\rm{CH}_3$', linewidth=6)
plt.plot(beta*oh_dat[:,0], oh_dat[:,2], 'b-', label=r'$\rm{OH}$', linewidth=6)

plt.xlabel(r'$\beta \phi$')
plt.ylabel(r'$- \frac{d\langle N_V \rangle_{\phi}}{d \beta \phi}$')

plt.xlim(0,beta*5)
plt.xticks([0,1,2])
plt.ylim(0,222)

plt.legend(loc=1)
plt.tight_layout()
plt.savefig('/Users/nickrego/Desktop/sam_sus_v_phi.pdf')
plt.show()