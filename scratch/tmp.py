import os, glob
import pymbar

def plt_errorbars(bb, vals, errs, **kwargs):
    ax = plt.gca()
    ax.plot(bb, vals)
    ax.fill_between(bb, vals-errs, vals+errs, alpha=0.5, **kwargs)


def bootstrap(dat):
    np.random.seed()

    iact = np.ceil(pymbar.timeseries.integratedAutocorrelationTime(dat))
    n_eff = int(dat.size / (1+2*iact))

    n_boot = 1000
    boot = np.zeros(n_boot)

    for i in range(1000):
        this_boot_sample = np.random.choice(dat, size=n_eff, replace=True)
        boot[i] = this_boot_sample.mean()


    return boot.std(ddof=1)

def get_avg_box_vec(univ, start=500):

    n_frames = univ.trajectory.n_frames - start
    box_v = np.zeros(3)

    for i in range(start, univ.trajectory.n_frames):
        univ.trajectory[i]
        box_v += univ.dimensions[:3]

    box_v /= n_frames

    return box_v

bGetbphi = True

# Compare nvphi for old and new gromacs indus
homedir = os.environ['HOME']

dat = np.loadtxt('new1/NvPhi.dat')
bphi = dat[:,0]

new_nvphi = np.zeros((dat.shape[0], 3))
old_nvphi = np.zeros_like(new_nvphi)

new_bphi0 = np.zeros(3)
old_bphi0 = np.zeros(3)
new_bphistar = np.zeros(3)
old_bphistar = np.zeros(3)

new_box_bphi0 = np.zeros((3, 3))
old_box_bphi0 = np.zeros((3, 3))

new_box_bphistar = np.zeros((3, 3))
old_box_bphistar = np.zeros((3, 3))


for i in range(1,4):
    newfile = np.loadtxt('new{}/NvPhi.dat'.format(i))
    oldfile = np.loadtxt('old{}/NvPhi.dat'.format(i))

    assert np.array_equal(bphi, newfile[:,0])
    assert np.array_equal(bphi, oldfile[:,0])

    new_nvphi[:,i-1] = newfile[:,1]
    old_nvphi[:,i-1] = oldfile[:,1]

    if bGetbphi:
        new_bphi0[i-1] = dr.loadPhi('new{}/beta_phi_000/phiout.dat'.format(i)).data[500:]['$\~N$'].mean()
        old_bphi0[i-1] = dr.loadPhi('old{}/beta_phi_000/phiout.dat'.format(i)).data[500:]['$\~N$'].mean()

        new_bphistar[i-1] = dr.loadPhi('new{}/beta_phi_star/phiout.dat'.format(i)).data[500:]['$\~N$'].mean()
        old_bphistar[i-1] = dr.loadPhi('old{}/beta_phi_star/phiout.dat'.format(i)).data[500:]['$\~N$'].mean()

        new_box_bphi0[i-1] = get_avg_box_vec(MDAnalysis.Universe('new{}/beta_phi_000/confout.gro'.format(i), 'new{}/beta_phi_000/traj.trr'.format(i)))
        old_box_bphi0[i-1] = get_avg_box_vec(MDAnalysis.Universe('old{}/beta_phi_000/confout.gro'.format(i), 'old{}/beta_phi_000/traj.trr'.format(i)))

        new_box_bphistar[i-1] = get_avg_box_vec(MDAnalysis.Universe('new{}/beta_phi_star/confout.gro'.format(i), 'new{}/beta_phi_star/traj.trr'.format(i)))
        old_box_bphistar[i-1] = get_avg_box_vec(MDAnalysis.Universe('old{}/beta_phi_star/confout.gro'.format(i), 'old{}/beta_phi_star/traj.trr'.format(i)))



avg_new = new_nvphi.mean(axis=1)
avg_old = old_nvphi.mean(axis=1)

err_new = new_nvphi.std(axis=1, ddof=1)
err_old = old_nvphi.std(axis=1, ddof=1)


