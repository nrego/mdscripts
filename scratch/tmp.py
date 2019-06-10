import os, glob
from util import assign_and_average
from scipy.integrate import cumtrapz

homedir = os.environ['HOME']
gamma = 50
fnames = sorted(glob.glob('umbr/nstar*'))

nstars = []

f_k = np.loadtxt('umbr/f_k_all.dat')
avg_ns = []
plt.close('all')
for fname in fnames:
    ds = dr.loadPhi('{}/phiout.dat'.format(fname))
    avg_n = ds.data[1000:]['N'].mean()
    avg_ns.append(avg_n)
    is_neg = fname.split('_')[1] == 'neg'
    num = float(fname.split('_')[-1]) 
    num = -num if is_neg else num

    nstars.append(num)

nstars = np.array(nstars)
avg_ns = np.array(avg_ns)
f_k -= f_k.min()

sort_idx = np.argsort(nstars)
mask = nstars[sort_idx] > -50

dt = 0.1
start = 1000
end = 10000

# tamd dat
dat = np.loadtxt('tamd_gam_{}/phiout_tamd.dat'.format(gamma))[int(start/dt):]
kappa = 0.420
beta = 1 / (k*300)

ntwid = dat[:,2]
nstar = dat[:,3]
min_pt = np.floor(nstar.min())
max_pt = np.ceil(nstar.max())
bins = np.arange(min_pt, max_pt+1, 0.1)

forces = beta*kappa*(ntwid-nstar)
bin_assign = np.digitize(nstar, bins) - 1

avg_force_by_kappa = assign_and_average(forces.astype(np.float32), bin_assign, bins.size-1)
masked_ar = np.ma.masked_invalid(avg_force_by_kappa)
avg_force_by_kappa[masked_ar.mask] = 0

integ = cumtrapz(-avg_force_by_kappa, bins[:-1])
integ -= integ.min()
f_k -= f_k.min()

fig, ax = plt.subplots(figsize=(7,6))
ax.plot(bins[:-2], integ, label='tamd, gamma={}'.format(gamma))
ax.plot(nstars[sort_idx][mask], f_k[sort_idx][mask], 'x', label='umbrella')
ax.legend()

ax.set_xlabel(r'$N^*$')
ax.set_ylabel(r'$\beta F_{\kappa, N^*}$')
fig.tight_layout()

plt.savefig('{}/Desktop/gam_{}'.format(homedir, gamma), transparent=True)
outdat = np.hstack((bins[:-2, None], integ[:, None]))
np.savetxt('tamd_gam_{}/f_k.dat'.format(gamma), outdat)
