from scipy.spatial import cKDTree
import numpy as np
import MDAnalysis
import glob, os, pathlib
from scipy.integrate import cumtrapz

fnames = sorted(glob.glob('*/prot_contact/noq_phi_sims/PvN.dat'))
homedir = os.environ['HOME']

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

    rho_noq = np.load('{}/noq_phi_sims/equil/rho_data_dump_rad_6.0.dat.npz'.format(topdir))['rho_water'].mean(axis=0)
    rho_q = np.load('{}/phi_sims/equil/rho_data_dump_rad_6.0.dat.npz'.format(topdir))['rho_water'].mean(axis=0)

    univ = MDAnalysis.Universe('{}/phi_sims/cent.gro'.format(topdir))
    prot = univ.select_atoms('protein and not name H*')
    prot = univ.select_atoms("not resname SOL and not name H* and not name CL and not name NA")
    assert prot.n_atoms == rho_noq.size == rho_q.size
    
    univ.add_TopologyAttr('tempfactors')
    contact_mask = np.loadtxt('{}/old_prot_all/bound/actual_contact_mask.dat'.format(path.parts[0]), dtype=bool)
    buried_mask = np.loadtxt('{}/old_prot_all/bound/buried_mask.dat'.format(path.parts[0]), dtype=bool)
    #prot.tempfactors = -2
    #prot[contact_mask].tempfactors = (rho_noq/rho_q)[contact_mask]
    prot.tempfactors = (rho_noq/rho_q)
    prot[buried_mask].tempfactors = -2
    prot.write('{}/Desktop/prot_{}.pdb'.format(homedir,name), bonds=None)


    plt.close('all')
    fig = plt.figure(figsize=(7,6))
    ax = fig.gca()
    ax.plot(dat_q[:,0], dat_q[:,1], label='{}, reg'.format(name))
    ax.plot(dat_noq[:,0], dat_noq[:,1], label='{}, q=0'.format(name))
    ax.set_xlabel(r'$N$')
    ax.set_ylabel(r'$\beta F_v(N)$')
    fig.legend()
    fig.tight_layout()
    fig.savefig('{}/Desktop/fig_fvn_{}'.format(homedir, name), transparent=True)


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

    n0_noq = n_noq[0,1]
    n0_q = n_q[0,1]

    print('Name: {}  <Nv>0_reg: {:1.2f}   <Nv>0_noq: {:1.2f}  Delta <Nv>0: {:1.2f}  ({:0.2f})'.format(name, n0_q, n0_noq, n0_q-n0_noq, (n0_q-n0_noq)/n0_q))


    plt.close('all')
    fig = plt.figure(figsize=(7,6))
    ax = fig.gca()
    ax.plot(n_q[:,0], n_q[:,1], label='{}, reg'.format(name))
    ax.plot(n_noq[:,0], n_noq[:,1], label='{}, q=0'.format(name))
    ax.set_xlabel(r'$\beta \phi$')
    ax.set_ylabel(r'$\langle N_v \rangle_\phi$')
    fig.legend()
    fig.tight_layout()
    fig.savefig('{}/Desktop/fig_nvphi_{}'.format(homedir, name), transparent=True)




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


