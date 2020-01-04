import os, glob
from util import assign_and_average
from scipy.integrate import cumtrapz

def plot_errorbar(bb, dat, err, **kwargs):
    plt.plot(bb, dat, **kwargs)
    plt.fill_between(bb, dat-err, dat+err, alpha=0.5)

pvn = np.loadtxt("PvN.dat")
nvphi = np.loadtxt("NvPhi.dat")

ntwid_dat = np.loadtxt("ntwid_out.dat")
ntwid_err = np.loadtxt("ntwid_err.dat")
nvar_err = np.loadtxt("ntwid_var_err.dat")

plt.close('all')
plot_errorbar(pvn[:,0], pvn[:,1], pvn[:,2])
plt.xlabel(r'$\tilde{N}$')
plt.ylabel(r'$-\ln P_v(\tilde{N})$')
plt.tight_layout()

plt.savefig('/home/nick/Desktop/pvn.png')

plt.close('all')
plot_errorbar(nvphi[:,0], nvphi[:,1], nvphi[:,3])
plt.errorbar(ntwid_dat[:,0], ntwid_dat[:,1], yerr=ntwid_err, fmt='o')
plt.xlabel(r'$\beta \phi$')
plt.ylabel(r'$\langle \tilde{N} \rangle_\phi$')
plt.xlim(-1,3)
plt.tight_layout()
plt.savefig('/home/nick/Desktop/nvphi.png')

plt.close('all')
plot_errorbar(nvphi[:,0], nvphi[:,2], nvphi[:,4])
plt.errorbar(ntwid_dat[:,0], ntwid_dat[:,2], yerr=nvar_err, fmt='o')
plt.xlabel(r'$\beta \phi$')
plt.ylabel(r'$\langle \tilde{\delta N^2} \rangle_\phi$')
plt.xlim(-1,3)
plt.tight_layout()
plt.savefig('/home/nick/Desktop/susvphi.png')

