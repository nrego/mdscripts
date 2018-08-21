
import numpy as np
from whamutils import get_neglogpdist


boot_dat = np.load('boot_fn_payload.dat.npy')
n_boot = boot_dat.shape[0]

logweights_0, dat_0, dat_N_0 = boot_dat[0]
bb = np.arange(0,200,1.)

boot_neglogpdist = np.zeros((n_boot, bb.size-1))

for i, payload in enumerate(boot_dat):

    logweights, all_dat, all_dat_N = payload

    neglogpdist = get_neglogpdist(all_dat_N.astype(float), bb, logweights)
    boot_neglogpdist[i] = neglogpdist


avg_neglogpdist = boot_neglogpdist.mean(axis=0)

for bphi in np.arange(1.0, 1.4, 0.1):
    bias = avg_neglogpdist + bphi*bb[:-1]
    bias -= bias.min()

    plt.plot(bb[:-1], bias, label='{}'.format(bphi))

plt.legend()
plt.show()

