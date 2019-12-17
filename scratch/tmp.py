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


# Compare nvphi for old and new gromacs indus
homedir = os.environ['HOME']

dat = np.loadtxt('new1/NvPhi.dat')
bphi = dat[:,0]

new_nvphi = np.zeros((dat.shape[0], 3))
old_nvphi = np.zeros_like(new_nvphi)

## get beta phis
'''
new_0 = np.array(dr.loadPhi('new1/beta_phi_000/phiout.dat').data['$\~N$'][500:])
old_0 = np.array(dr.loadPhi('old1/beta_phi_000/phiout.dat').data['$\~N$'][500:])
err_new_0 = bootstrap(new_0)
err_old_0 = bootstrap(old_0)

if err_new_0 + err_old_0 > np.abs(new_0.mean() - old_0.mean()):
    print('\nbphi=0 NOT sig different\n')
else:
    print('\nbphi=0 sig different!\n')

new_phistar = dr.loadPhi('new1/beta_phi_200/phiout.dat').data['$\~N$'][500:]
old_phistar = dr.loadPhi('old1/beta_phi_200/phiout.dat').data['$\~N$'][500:]
err_new_phistar = bootstrap(new_phistar)
err_old_phistar = bootstrap(old_phistar)

if err_new_phistar + err_old_phistar > np.abs(new_0.mean() - old_0.mean()):
    print('\nbphistar NOT sig different\n')
else:
    print('\nbphistar sig different!\n')
'''

for i in range(1,4):
    newfile = np.loadtxt('new{}/NvPhi.dat'.format(i))
    oldfile = np.loadtxt('old{}/NvPhi.dat'.format(i))

    assert np.array_equal(bphi, newfile[:,0])
    assert np.array_equal(bphi, oldfile[:,0])

    new_nvphi[:,i-1] = newfile[:,1]
    old_nvphi[:,i-1] = oldfile[:,1]

avg_new = new_nvphi.mean(axis=1)
avg_old = old_nvphi.mean(axis=1)

err_new = new_nvphi.std(axis=1, ddof=1)
err_old = old_nvphi.std(axis=1, ddof=1)


