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

n_vals = None
tmp_pvn_q_all = []
tmp_pvn_noq_all = []
names = []
for i, fname in enumerate(fnames):

    path = pathlib.Path(fname)
    name = name_lut[path.parts[0]]
    dirname = os.path.dirname(fname)
    topdir = os.path.dirname(dirname)
    names.append(name)

    dat_noq = np.loadtxt(fname)
    dat_q = np.loadtxt('{}/phi_sims/PvN.dat'.format(topdir))

    tmp_pvn_q_all.append(dat_q[:,1])
    tmp_pvn_noq_all.append(dat_noq[:,1])

    assert dat_noq[:,0].min() == dat_q[:,0].min() == 0
    assert dat_q[:,0].size >= dat_noq[:,0].size

    n_noq = np.loadtxt('{}/NvPhi.dat'.format(dirname))
    n_q = np.loadtxt('{}/phi_sims/NvPhi.dat'.format(topdir))

    if n_vals is None:
        n_vals = dat_q[:,0]
    elif n_vals.size < dat_q[:,0].size:
        n_vals = dat_q[:,0]

names = np.array(names)

pvn_q_all = np.zeros((n_vals.size, len(fnames)+1))
pvn_noq_all = np.zeros_like(pvn_q_all)
pvn_q_all[:] = np.inf
pvn_noq_all[:] = np.inf

pvn_q_all[:,0] = n_vals
pvn_noq_all[:,0] = n_vals

for i, (this_pvn_q, this_pvn_noq) in enumerate(zip(tmp_pvn_q_all, tmp_pvn_noq_all)):
    pvn_q_all[:this_pvn_q.size, i+1] = this_pvn_q
    pvn_noq_all[:this_pvn_noq.size, i+1] = this_pvn_noq

np.savetxt('fvn_q.dat', pvn_q_all, header='    '.join(names), fmt='%1.4e')
np.savetxt('fvn_noq.dat', pvn_noq_all, header='    '.join(names), fmt='%1.4e')


for i, name in enumerate(names):
    this_pvn_q = pvn_q_all[:, i+1]
    this_pvn_noq = pvn_noq_all[:, i+1]

    n0_q = n_vals[np.argmin(this_pvn_q)]
    n0_noq = n_vals[np.argmin(this_pvn_noq)]

    delta_n_q = n0_q - n_vals
    delta_n_noq = n0_noq - n_vals

    plt.plot(delta_n_noq, this_pvn_noq/delta_n_noq, label=name)

plt.legend()
plt.show()


