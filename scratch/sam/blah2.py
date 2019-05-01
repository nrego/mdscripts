import os, glob
from scipy import special

fnames = sorted(glob.glob('density_*'))

bins = None
omega_tot = None

homedir = os.environ['HOME']

from constants import k

avg_es = []
var_es = []
k_vals = []
for fname in fnames:
    ds = np.load(fname)
    if bins is None:
        bins = ds['bins']
    else:
        assert np.array_equal(bins, ds['bins'])
    if omega_tot is None:
        omega_tot = ds['omega']
    else:
        omega_tot += ds['omega']

    k_ch3 = int(ds['k_ch3'])

    this_omega = ds['omega']
    this_prob = this_omega / special.binom(36, k_ch3)

    avg_e = np.dot(bins[:-1], this_prob)
    avg_esq = np.dot(bins[:-1]**2, this_prob)

    var_e = avg_esq - avg_e**2

    occ = this_omega > 0
    min_e = bins[:-1][occ].min()
    max_e = bins[:-1][occ].max()
    print('K_C: {:d}'.format(int(ds['k_ch3'])))
    print('  energy range: {:.2f} to {:.2f} ({:.2f})'.format(min_e, max_e, max_e-min_e))
    print('  avg e: {:0.2f}  var_e: {:0.2f}'.format(avg_e, var_e))
    k_vals.append(k_ch3)
    avg_es.append(avg_e)
    var_es.append(var_e)

k_vals = np.array(k_vals)
avg_es = np.array(avg_es)
var_es = np.array(var_es)

entropy = k*np.log(omega_tot)
mask = ~np.ma.masked_invalid(entropy).mask

errs = []
for i in range(6):
    coef = np.polyfit(bins[:-1][mask], entropy[mask], deg=i)
    p = np.poly1d(coef)
    fit = p(bins[:-1])
    err = fit[mask] - entropy[mask]
    mse = np.mean(err**2)
    print("i: {} Mse: {:1.2e}".format(i, mse))
    errs.append(np.sqrt(mse))

coef = np.polyfit(bins[:-1][mask], entropy[mask], deg=5)
p = np.poly1d(coef)

evals = np.arange(135,287,0.01)

plt.plot(evals, p(evals))

plt.plot(bins[:-1], entropy, 'x')


fig, ax = plt.subplots()
pprime = p.deriv()

ax.plot(evals, 1/pprime(evals))
ax.set_ylim(-10, 90)
ax.set_yticks([0,300,600,900])
fig.tight_layout()
fig.savefig('/Users/nickrego/Desktop/fig.png')
