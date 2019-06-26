from util import assign_and_average
from whamutils import gen_data_logweights
from scipy.integrate import cumtrapz

Ub = lambda n, nstar, beta, kappa: beta*kappa*(n - nstar)**2

gamma = 20

# tamd dat
start = 1000
dt = 0.1
dat = np.loadtxt('new_tamd_gam_{}/phiout_tamd.dat'.format(gamma))[int(start/dt):]
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
f_k = -cumtrapz(avg_force_by_kappa, bins[:-1])
f_k -= f_k.min()

nstar_vals = bins[:-2]
bin_assign_ntwid = np.digitize(ntwid, bins[:-1]) - 1
bin_assign_nstar = np.digitize(ntwid, bins[:-1]) - 1
n_k = np.zeros_like(nstar_vals)
for i in range(n_k.size):
    n_k[i] = (bin_assign_nstar == i).sum()
n_k /= n_k.sum()

xx, yy = np.meshgrid(nstar_vals, nstar_vals, indexing='ij')
bias_mat = Ub(xx, yy, beta, kappa)

logweights = gen_data_logweights(bias_mat, f_k, n_k)

max_logweight = logweights.max()
logweights -= max_logweight
norm = max_logweight + np.log(np.exp(logweights).sum())
logweights -= norm

f = -logweights

outofbounds_mask = (nstar_vals < ntwid.min()) | (nstar_vals > ntwid.max())
f[outofbounds_mask] = np.inf
f -= f.min()

plt.plot(nstar_vals, f_k)
plt.plot(nstar_vals, f)

