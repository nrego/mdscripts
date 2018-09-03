import numpy as np
from whamutils import get_neglogpdist
from constants import k

kt = k*300
beta = 1/kt

def plt_errorbars(bb, loghist, errs):
    plt.fill_between(bb[:-1], loghist-errs, loghist+errs, alpha=0.5)


def logsumexp(data, logweights):
    maxpt = logweights.max()
    logweights -= maxpt

    return np.dot(np.exp(logweights), data) + maxpt
    
def get_avgs(data, logweights, phi_vals):
    pass

boot_dat = np.load('boot_fn_payload.dat.npy')
n_boot = boot_dat.shape[0]

logweights_0, dat_0, dat_N_0 = boot_dat[0]

bb = np.arange(0, 2001, 1).astype(float)
max_N = 0
phi_vals = np.arange(0, 10.1, 0.1) * beta

boot_avg = np.zeros((n_boot, len(phi_vals)))
boot_var = np.zeros_like(boot_avg)

boot_neglogpdist = np.zeros((n_boot, bb.size-1))
boot_neglogpdist_N = np.zeros_like(boot_neglogpdist)

for i, payload in enumerate(boot_dat):
    #print("doing {}".format(i))
    logweights, all_dat, all_dat_N = payload

    max_N = np.ceil(max(max_N, np.max((all_dat, all_dat_N)))) + 1
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

avg_mean = boot_avg.mean(axis=0)
avg_err = boot_avg.std(axis=0, ddof=1)

var_mean = boot_var.mean(axis=0)
var_err = boot_avg.std(axis=0, ddof=1)

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



