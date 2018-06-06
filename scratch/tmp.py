import numpy as np


boot_dat = np.load('boot_fn_payload.dat.npy')
n_boot = boot_dat.shape[0]

neglogpdist0, bb0, dat_arr0 = boot_dat[0]
n_phi_vals = dat_arr0.shape[0]

phi_vals = dat_arr0[:,0]

boot_neglogpdist = np.zeros((n_boot, neglogpdist0.size))
boot_avg_n = np.zeros((n_boot, n_phi_vals))
boot_var_n = np.zeros((n_boot, n_phi_vals))

for i, payload in enumerate(boot_dat):

    neglogpdist, bb, dat_arr = payload

    np.testing.assert_array_equal(dat_arr[:,0], phi_vals)

    boot_neglogpdist[i] = neglogpdist
    boot_avg_n[i] = dat_arr[:,1]
    boot_var_n[i] = dat_arr[:,2]

avg_n = boot_avg_n.mean(axis=0)
err_avg_n = boot_avg_n.std(axis=0, ddof=1)

var_n = boot_var_n.mean(axis=0)
err_var_n = boot_var_n.std(axis=0, ddof=1)

out_dat_arr = np.vstack((phi_vals, avg_n, var_n)).T
err_out_dat_arr = np.vstack((err_avg_n, err_var_n)).T

np.savetxt('n_v_phi.dat', out_dat_arr)
np.savetxt('err_n_v_phi.dat', err_out_dat_arr)
