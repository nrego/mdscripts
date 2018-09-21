import numpy as np
from whamutils import get_neglogpdist
from constants import k

kt = k*300
beta = 1/kt

# shape: (n_boot_samples, 3)
#    col0:  logweight of each bootstrap sample (log of UWHAM weight)
#    col1:  Ntwid_v for each 
#    col2:  N_v for each
boot_dat = np.load('boot_fn_payload.dat.npy')
n_boot = boot_dat.shape[0]

logweights_0, dat_0, dat_N_0 = boot_dat[0]

# binbounds for P_v(N)'s
bb = np.arange(0, 2001, 1).astype(float)
max_N = 0

# beta*phi values of linear bias for re-weighting to get <N_v>phi v phi
phi_vals = np.arange(0, 4.1, 0.001) #* beta

# Bootstraps of <N_v>_phi  and <d N_v^2>_phi
boot_avg = np.zeros((n_boot, len(phi_vals)))
boot_var = np.zeros_like(boot_avg)

# Bootstraps of -ln P_v(Ntwid) and -ln P_v(N)
boot_neglogpdist = np.zeros((n_boot, bb.size-1))
boot_neglogpdist_N = np.zeros_like(boot_neglogpdist)

# beta phi* for each bootstrap
boot_peak_sus = np.zeros(n_boot)

for i, payload in enumerate(boot_dat):

    logweights, all_dat, all_dat_N = payload

    neglogpdist = get_neglogpdist(all_dat, bb, logweights)
    neglogpdist_N = get_neglogpdist(all_dat_N.astype(float), bb, logweights)

    boot_neglogpdist[i] = neglogpdist
    boot_neglogpdist_N[i] = neglogpdist_N

    for idx, phi in enumerate(phi_vals):
        bias = phi * all_dat
        this_weights = logweights - bias
        this_weights -= this_weights.max()
        this_weights = np.exp(this_weights)
        this_weights /= this_weights.sum()

        this_avg_n = np.dot(this_weights, all_dat)
        this_avg_nsq = np.dot(this_weights, all_dat**2)
        this_var_n = this_avg_nsq - this_avg_n**2

        boot_avg[i,idx] = this_avg_n
        boot_var[i,idx] = this_var_n

    max_idx = np.argmax(boot_var[i])
    boot_peak_sus[i] = phi_vals[max_idx]

avg_mean = boot_avg.mean(axis=0)
avg_err = boot_avg.std(axis=0, ddof=1)

var_mean = boot_var.mean(axis=0)
var_err = boot_var.std(axis=0, ddof=1)

np.savetxt('n_v_phi.dat', np.vstack((phi_vals, avg_mean, avg_err)).T)
np.savetxt('var_n_v_phi.dat', np.vstack((phi_vals, var_mean, var_err)).T)


masked_neglogpdist = np.ma.masked_invalid(boot_neglogpdist)
masked_neglogpdist_N = np.ma.masked_invalid(boot_neglogpdist_N)

avg_neglogpdist = masked_neglogpdist.mean(axis=0)
err_neglogpdist = masked_neglogpdist.std(axis=0, ddof=1)

mask = avg_neglogpdist.mask
avg_neglogpdist = avg_neglogpdist.data
avg_neglogpdist[mask] = np.float('inf')
err_neglogpdist = err_neglogpdist.data
err_neglogpdist[mask] = np.float('inf')

avg_neglogpdist_N = masked_neglogpdist_N.mean(axis=0)
err_neglogpdist_N = masked_neglogpdist_N.std(axis=0, ddof=1)

mask_N = avg_neglogpdist_N.mask
avg_neglogpdist_N = avg_neglogpdist_N.data
avg_neglogpdist_N[mask] = np.float('inf')
err_neglogpdist_N = err_neglogpdist_N.data
err_neglogpdist_N[mask] = np.float('inf')

np.savetxt('neglogpdist.dat', np.vstack((bb[:-1], avg_neglogpdist, err_neglogpdist)).T)
np.savetxt('neglogpdist_N.dat', np.vstack((bb[:-1], avg_neglogpdist_N, err_neglogpdist_N)).T)

avg_peak_sus = boot_peak_sus.mean()
err_peak_sus = boot_peak_sus.std(ddof=1)
np.savetxt('peak_sus.dat', [avg_peak_sus, err_peak_sus])
