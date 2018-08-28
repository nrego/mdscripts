
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
bb = np.arange(0,2000,1.)

boot_neglogpdist = np.zeros((n_boot, bb.size-1))
boot_neglogpdist_N = np.zeros_like(boot_neglogpdist)
boot_fprime = np.zeros((n_boot, bb.size-2))

max_N = 0
phi_vals = np.arange(0, 10.1, 0.1) * beta

for i, payload in enumerate(boot_dat):
    #print("doing {}".format(i))
    logweights, all_dat, all_dat_N = payload

    max_N = np.ceil(max(max_N, np.max((all_dat, all_dat_N)))) + 1

    neglogpdist = get_neglogpdist(all_dat.astype(float), bb, logweights)
    neglogpdist_N = get_neglogpdist(all_dat_N.astype(float), bb, logweights)

    boot_neglogpdist[i] = neglogpdist
    boot_neglogpdist_N[i] = neglogpdist_N

range_mask = bb < max_N

bb = bb[range_mask]
range_mask = range_mask[:-1]

boot_neglogpdist = np.ma.masked_invalid(boot_neglogpdist[:, range_mask])
avg_neglogpdist = boot_neglogpdist.mean(axis=0)
err_neglogpdist = boot_neglogpdist.std(axis=0, ddof=1)
mask = avg_neglogpdist.mask
avg_neglogpdist.data[mask] = 'inf'

outarr = np.vstack((bb, avg_neglogpdist, err_neglogpdist)).T
np.savetxt('neglogpdist.dat', outarr)

boot_neglogpdist_N = np.ma.masked_invalid(boot_neglogpdist_N[:, range_mask])
avg_neglogpdist_N = boot_neglogpdist_N.mean(axis=0)
err_neglogpdist_N = boot_neglogpdist_N.std(axis=0, ddof=1)
mask = avg_neglogpdist_N.mask
avg_neglogpdist_N.data[mask] = 'inf'

outarr_N = np.vstack((bb, avg_neglogpdist_N, err_neglogpdist_N)).T
np.savetxt('neglogpdist_N.dat', outarr_N)

