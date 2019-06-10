import os, glob
from util import assign_and_average
from scipy.integrate import cumtrapz

homedir = os.environ['HOME']
gamma = 10

def get_fk(ntwid, nstar, bins, kappa, beta):
    forces = beta*kappa*(ntwid-nstar)
    bin_assign = np.digitize(nstar, bins) - 1

    avg_force_by_kappa = assign_and_average(forces.astype(np.float32), bin_assign, bins.size-1)
    masked_ar = np.ma.masked_invalid(avg_force_by_kappa)
    avg_force_by_kappa[masked_ar.mask] = 0

    f_k = cumtrapz(-avg_force_by_kappa, bins[:-1])
    f_k -= f_k.min()
    f_k[masked_ar.mask[:-1]] = np.inf

    return f_k

dt = 0.1
start = 100
end = 10000

# tamd dat
dat = np.loadtxt('tamd_gam_{}/phiout_tamd.dat'.format(gamma))
kappa = 0.420
beta = 1 / (k*300)

ntwid = dat[:,2]
nstar = dat[:,3]
min_pt = np.floor(nstar.min())
max_pt = np.ceil(nstar.max())
bins = np.arange(min_pt, max_pt+1, 0.1)

plt.close('all')
fig, ax = plt.subplots(figsize=(7,6))
cut_dat = dat[int(start/dt):]
ax.plot(cut_dat[:,0], cut_dat[:,2], label=r'$N^*$')
ax.plot(cut_dat[:,0], cut_dat[:,3], label=r'$\tilde{N}$')
ax.set_xlabel(r'$t$ (ps)')
fig.tight_layout()
plt.legend()
fig.savefig('{}/Desktop/gam_{}_traj.png'.format(homedir, gamma))


fig, ax = plt.subplots(figsize=(6,5))
for i, endpt in enumerate([5000, 10000, 25000, 50000]):
    subdat = dat[int(start/dt):int(endpt/dt)]
    ntwid = subdat[:,2]
    nstar = subdat[:,3]

    f_k = get_fk(ntwid, nstar, bins, kappa, beta)
    ax.plot(bins[:-2], f_k, label=r'${:0.0f}$ ns'.format(endpt/1000), zorder=4-i, linewidth=4)
plt.legend()
ax.set_xlabel(r'$N^*$')
ax.set_ylabel(r'$\beta F_{\kappa, N^*}$')
fig.tight_layout()
fig.savefig('{}/Desktop/gamma_{}_conv'.format(homedir, gamma), transparent=True)


