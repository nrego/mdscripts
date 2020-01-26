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
    '2tsc_pred': 'TS'
}

bphi = None
for i, fname in enumerate(fnames):

    path = pathlib.Path(fname)
    name = name_lut[path.parts[0]]
    dirname = os.path.dirname(fname)
    topdir = os.path.dirname(dirname)

    n0 = np.loadtxt('{}/NvPhi.dat'.format(dirname))[0,1]

    dat = np.loadtxt(fname)
    dat = np.loadtxt('{}/phi_sims/PvN.dat'.format(topdir))
    n = dat[:,0]
    beta_G_N = dat[:,1]

    n0 = n[np.argmin(beta_G_N)]

    n_diff = n0 - n
    mask = n_diff > 0
    plt.plot(n_diff[mask], (beta_G_N/n_diff)[mask], label=name)


plt.legend()
plt.show()