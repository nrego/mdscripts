from scipy.spatial import cKDTree
import numpy as np
import MDAnalysis
import glob, os, pathlib
from scipy.integrate import cumtrapz

fnames = sorted(glob.glob('*/prot_contact/noq_phi_sims/PvN.dat'))


name_lut = {
    '1bmd': 'MDH',
    '1brs_bn': 'barnase',
    '1brs_bs': 'barstar',
    '1msb_pred': 'MBP',
    '1ycr_mdm2': 'MDM2',
    '2mlt': 'MLT',
    '2tsc_pred': 'TS',
    '2z59_ubiq': 'UBQ'
}

bphi = None
for i, fname in enumerate(fnames):

    path = pathlib.Path(fname)
    name = name_lut[path.parts[0]]
    dirname = os.path.dirname(fname)
    topdir = os.path.dirname(dirname)

    dat_noq = np.loadtxt(fname)
    dat_q = np.loadtxt('{}/phi_sims/PvN.dat'.format(topdir))

    n0_noq = np.loadtxt('{}/NvPhi.dat'.format(dirname))[0,1]
    n0_q = np.loadtxt('{}/phi_sims/NvPhi.dat'.format(topdir))[0,1]

    n_noq = dat_noq[:,0]
    n_q = dat_q[:,0]
    assert n_noq[0] == n_q[0] == 0
    assert n_q.size >= n_noq.size

    f_q = dat_q[:,1]
    f_noq = np.zeros_like(f_q)
    f_noq[:] = np.nan
    f_noq[:n_noq.size] = dat_noq[:n_noq.size, 1]

    diff_f = f_q - f_noq
    print("{};  n0 (q): {:0.1f}  n0 (noq): {:0.1f}  diff: {:0.1f}".format(name, n0_q, n0_noq, n0_q - n0_noq))

    delta_n_noq = n0_noq - n_q
    delta_n_q = n0_q - n_q
    mask = delta_n_q > 0

    plt.plot(n_q/n0_q, (f_q-f_noq)/n0_q, label=name)


plt.legend()
plt.show()

