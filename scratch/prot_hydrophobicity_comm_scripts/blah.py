
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

max_N = 0
phi_vals = np.arange(0, 10.1, 0.1) * beta

boot_avg = np.zeros((n_boot, len(phi_vals)))
boot_var = np.zeros_like(boot_avg)

for i, payload in enumerate(boot_dat):
    #print("doing {}".format(i))
    logweights, all_dat, all_dat_N = payload

    max_N = np.ceil(max(max_N, np.max((all_dat, all_dat_N)))) + 1

    for idx, phi in enumerate(phi_vals):
        bias = phi * all_dat
        this_weights = logweights - bias
        this_weights -= this_weights.max()
        this_weights = np.exp(this_weights)
        this_weights /= this_weights.sum()

        this_avg_n = np.dot(this_weights, all_dat_N)
        this_avg_nsq = np.dot(this_weights, all_dat_N**2)
        this_var_n = this_avg_nsq - this_avg_n**2

        boot_avg[i,idx] = this_avg_n
        boot_var[i,idx] = this_var_n

avg_mean = boot_avg.mean(axis=0)
avg_err = boot_avg.std(axis=0, ddof=1)

var_mean = boot_var.mean(axis=0)
var_err = boot_avg.std(axis=0, ddof=1)

plt.savetxt('n_v_phi.dat', np.vstack((phi_vals, avg_mean, avg_err)))
plt.savetxt('var_n_v_phi.dat', np.vstack((phi_vals, var_mean, var_err)))
