import os, glob
from scipy import special
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams.update({'axes.labelsize': 40})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':10})

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
    plt.close('all')
    fig, ax = plt.subplots(figsize=(7,6))
    ax.plot(bins[:-1], k*np.log(this_omega))
    ax.set_xlim(135, 287)
    plt.tight_layout()
    plt.savefig('{}/Desktop/s_k_{:d}.png'.format(homedir, k_ch3))
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

plt.close('all')
fig, ax1 = plt.subplots(figsize=(7,6))
ax1.plot(k_vals, avg_es, '-bo', markersize=8, linewidth=3)
ax1.tick_params(axis='y', labelcolor='b')
ax2 = ax1.twinx()
ax2.plot(k_vals, var_es, '-ro', markersize=8, linewidth=3)
ax2.tick_params(axis='y', labelcolor='r')
ax1.set_xticks(np.arange(0,42,6))
plt.tight_layout()
plt.savefig('{}/Desktop/avg_var_with_k.png'.format(homedir))

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
plt.close('all')
fig, ax = plt.subplots(figsize=(7,6))

ax.plot(evals, p(evals))
ax.plot(bins[:-1], entropy, 'x')

ax.set_xlim(135,287)

plt.tight_layout()
plt.savefig('{}/Desktop/s_tot.png'.format(homedir))


