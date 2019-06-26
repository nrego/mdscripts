import os, glob
from scipy import special
import matplotlib as mpl
from matplotlib import pyplot as plt
import cPickle as pickle

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

def get_log_z(temp_vals, energies, entropy, quant=None):
    if quant is None:
        quant = np.ones_like(energies)
    betas = 1 / temp_vals
    # each will be shape: (n_energies, n_temps)
    bb, ee = np.meshgrid(betas, energies)
    neg_beta_e = -bb*ee

    expt = entropy[:,None] + neg_beta_e + np.log(quant[:,None])
    max_vals = expt.max(axis=0)
    mask = max_vals == -np.inf
    max_vals[mask] = 0
    expt -= max_vals

    return np.log(np.sum(np.exp(expt), axis=0)) + max_vals

def get_frac_kc(temp_vals, energies, entropy_by_kc):

    max_val = entropy_by_kc.max()
    total_entropy = np.log(np.exp(entropy_by_kc-max_val).sum(axis=0)) + max_val
    this_log_z = get_log_z(temp_vals, energies, total_entropy)
    total_avg_ko = np.zeros_like(temp_vals)
    for k_c in range(37):
        k_o = 36-k_c
        this_kc_entropy = entropy_by_kc[k_c]
        this_kc_weight = get_log_z(temp_vals, energies, this_kc_entropy, np.ones_like(this_kc_entropy)*k_o) - this_log_z
        this_weight = np.exp(this_kc_weight)
        if this_weight.min() < 0:
            embed()

        total_avg_ko += this_weight

    return total_avg_ko

def get_beta_fe_energy(temp, energies, entropy):
    beta = 1 / temp

    return beta*energies - entropy

with open('entropy_vals.pkl', 'r') as fin:
    payload = pickle.load(fin)

energies = payload['energies']
poly = payload['polynomial']
entropy = payload['entropy']
mask = ~np.ma.masked_invalid(entropy).mask
masked_energies = energies[~mask]
#print("masked energies: {}".format(masked_energies))
entropy = entropy[mask]
#entropy[0] = 0
energies = energies[mask]
e_min = energies.min()
energies -= e_min

sampled_entropy = payload['sampled_entropy']
sampled_energies = payload['sampled_energies']
sampled_entropy_by_kc = payload['sampled_entropy_by_kc']
sampled_energies -= e_min
pprime = poly.deriv()

ds_de = pprime(energies)


temp_vals = np.arange(0,100.1,0.01)
temp_vals = np.append(temp_vals, np.inf)
beta_vals = 1 / temp_vals
# -beta F(T), the log of the partition fn w/ T
log_z = get_log_z(temp_vals, energies, entropy)
beta_f = -log_z
# logarithm of energies (as fn of T)
log_e = get_log_z(temp_vals, energies, entropy, energies) - log_z
log_e_sq = get_log_z(temp_vals, energies, entropy, energies**2) - log_z
avg_e = np.exp(log_e)
var_e = np.exp(log_e_sq) - avg_e**2
# heat capacity
cv = var_e / temp_vals**2

avg_s = beta_vals*avg_e - beta_f

avg_e[0] = 0
cv[0] = 0
beta_f[0] = 0
avg_s[0] = 0


plt.close('all')
# Entropy vs energy
fig, ax = plt.subplots(figsize=(7,6))
ax.plot(energies, entropy)
ax.plot(sampled_energies, sampled_entropy, 'x')
ax.set_xlabel(r'$\hat{f}$')
ax.set_ylabel(r'$S(\hat{f}) = \ln g(\hat{f})$')
ymin, ymax = ax.get_ylim()
ax.set_ylim(0, ymax)
fig.tight_layout()

plt.savefig('{}/Desktop/comp_s_of_e'.format(homedir), transparent=True)

plt.close('all')
# Entropy vs energy
fig, ax = plt.subplots(figsize=(7,6))
ax.plot(energies, entropy)
ax.set_xlabel(r'$\hat{f}$')
ax.set_ylabel(r'$S(\hat{f}) = \ln g(\hat{f})$')
ymin, ymax = ax.get_ylim()
ax.set_ylim(0, ymax)
fig.tight_layout()

plt.savefig('{}/Desktop/s_of_e'.format(homedir), transparent=True)

plt.close('all')
# free energy, energy, entropy, and heat capacity vs (evolutionary, unitless) temperature
fig, ax = plt.subplots(figsize=(7,6))
ax.plot(temp_vals, beta_f, 'g--', label=r'$\beta F$')
ax.plot(temp_vals, avg_e, 'r--', label=r'$U=\langle \hat{f} \rangle$')
#ax.plot(temp_vals, var_e, 'k--', label=r'$\langle \delta \hat{f}^2 \rangle$')
ax.plot(temp_vals, cv, 'k-', label=r'$C_v$')
ax.plot(temp_vals, avg_s, 'b--', label=r'$S$')

plt.legend()
ymin, ymax = ax.get_ylim()
ax.set_xlim(0, 10)
ax.set_ylabel(r'$\beta F, U, C_v, S$')
ax.set_xlabel(r'$T$')
fig.tight_layout()

plt.savefig('{}/Desktop/thermo_with_temp'.format(homedir), transparent=True)

plt.close('all')
