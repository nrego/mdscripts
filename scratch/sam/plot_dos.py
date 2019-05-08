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

    this_omega = ds['omega'].astype(float)

    if omega_tot is None:
        omega_tot = this_omega.copy()
    else:
        omega_tot += this_omega

    k_ch3 = int(ds['k_ch3'])

    
    plt.close('all')
    #fig, ax = plt.subplots(figsize=(7,6))
    #ax.plot(bins[:-1], k*np.log(this_omega))
    #ax.set_xlim(bins.min(), bins.max())
    #plt.tight_layout()
    #plt.savefig('{}/Desktop/s_k_{:d}.png'.format(homedir, k_ch3))
    assert np.allclose(this_omega.sum(), special.binom(144, k_ch3))
    this_prob = this_omega / this_omega.sum()

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
for i in range(10):
    coef = np.polyfit(bins[:-1][mask], entropy[mask], deg=i)
    p = np.poly1d(coef)
    fit = p(bins[:-1])
    err = fit[mask] - entropy[mask]
    mse = np.mean(err**2)
    print("i: {} Mse: {:1.2e}".format(i, mse))
    errs.append(np.sqrt(mse))


## On energy
coef = np.polyfit(bins[:-1][mask], entropy[mask], deg=5)
p = np.poly1d(coef)
pprime = p.deriv()

evals = np.arange(135,287,0.01)
dos_fit = p(evals)
dos_fit[dos_fit<0] = np.nan

plt.close('all')
fig, ax = plt.subplots(figsize=(7,6))

ax.plot(evals, dos_fit)
ax.plot(bins[:-1], entropy, 'x')

ax.set_xlim(135,287)
ax.set_xlabel('$\hat{f}$')
ax.set_ylabel('$S(\hat{f})$')

plt.tight_layout()
plt.savefig('{}/Desktop/s_tot.png'.format(homedir), transparent=True)

## 
plt.close('all')
fig, ax = plt.subplots(figsize=(7,6))

dos_deriv = pprime(evals)
ax.plot(evals, 1/dos_deriv)

ax.set_xlim(135, 287)
ax.set_xlabel('$\hat{f}$')
ax.set_ylim(0, 500)
#ax.set_ylabel('$\frac{\partial S(\hat{f})}{\partial \hat{f}}$')

plt.tight_layout()
plt.savefig('{}/Desktop/temp.png'.format(homedir), transparent=True)

temps = np.arange(290, 315, 5)

prob_e = []
mask2 = ~np.ma.masked_invalid(dos_fit).mask
plt.close('all')
for temp in temps:
    beta = 1/(k*temp)
    beta_e = beta*evals
    exp = (dos_fit/k) - beta_e
    max_val = np.nanmax(exp)
    exp -= max_val
    prob = np.exp(exp)
    norm = np.trapz(prob[mask2], evals[mask2])
    prob /= norm
    fe = -np.log(prob)
    prob_e.append(fe) 

    plt.plot(evals, fe, label=r'$T={:d}$'.format(temp))


#########################
## On negative of energy
coef = np.polyfit(-bins[:-1][mask], entropy[mask], deg=5)
p = np.poly1d(coef)
pprime = p.deriv()

evals = -np.arange(135,287,0.01)
evals = evals[::-1]
dos_fit = p(evals)
dos_fit[dos_fit<0] = np.nan

plt.close('all')
fig, ax = plt.subplots(figsize=(7,6))

ax.plot(evals, dos_fit)
ax.plot(-bins[:-1], entropy, 'x')

ax.set_xlim(-287,-135)
ax.set_xlabel('$-\hat{f}$')
ax.set_ylabel('$S(-\hat{f})$')

plt.tight_layout()
plt.savefig('{}/Desktop/s_tot_rev.png'.format(homedir), transparent=True)

## 
plt.close('all')
fig, ax = plt.subplots(figsize=(7,6))

dos_deriv = pprime(evals)
ax.plot(evals, 1/dos_deriv)

ax.set_xlim(-287, -135)
ax.set_xlabel('$-\hat{f}$')
ax.set_ylim(0, 500)
#ax.set_ylabel('$\frac{\partial S(\hat{f})}{\partial \hat{f}}$')

plt.tight_layout()
plt.savefig('{}/Desktop/temp_rev.png'.format(homedir), transparent=True)

plt.close('all')

prob_e = []
mask2 = ~np.ma.masked_invalid(dos_fit).mask
for temp in temps:
    beta = 1/(k*temp)
    beta_e = beta*evals
    exp = (dos_fit/k) - beta_e
    max_val = np.nanmax(exp)
    exp -= max_val
    prob = np.exp(exp)
    norm = np.trapz(prob[mask2], evals[mask2])
    prob /= norm
    fe = -np.log(prob)
    prob_e.append(fe) 

    plt.plot(evals, fe, label=r'$T={:d}$'.format(temp))



