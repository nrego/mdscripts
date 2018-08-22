
import numpy as np
from whamutils import get_neglogpdist

def plt_errorbars(bb, loghist, errs):
    plt.fill_between(bb[:-1], loghist-errs, loghist+errs, alpha=0.5)

boot_dat = np.load('boot_fn_payload.dat.npy')
n_boot = boot_dat.shape[0]

logweights_0, dat_0, dat_N_0 = boot_dat[0]
bb = np.arange(0,2000,1.)

boot_neglogpdist = np.zeros((n_boot, bb.size-1))

for i, payload in enumerate(boot_dat):

    logweights, all_dat, all_dat_N = payload

    neglogpdist = get_neglogpdist(all_dat.astype(float), bb, logweights)
    boot_neglogpdist[i] = neglogpdist

nomask = boot_neglogpdist
boot_neglogpdist = np.ma.masked_invalid(boot_neglogpdist)
avg_neglogpdist = boot_neglogpdist.mean(axis=0)
err_neglogpdist = boot_neglogpdist.std(axis=0, ddof=1)
mask = avg_neglogpdist.mask
avg_neglogpdist.data[mask] = 'inf'
outarr = np.vstack((bb[:-1], avg_neglogpdist, err_neglogpdist)).T
np.savetxt('neglogpdist.dat', outarr)



