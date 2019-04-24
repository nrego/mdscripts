
from constants import k

kT = k * 300

ds = dr.loadPhi('phi_000/phiout.dat')
dat = np.array(ds.data[200:]['$\~N$'])

avg_n = dat.mean()
var_n = dat.var()

alpha = 3.

k = (alpha*kT) / (var_n)
n_min = int(np.floor(- (avg_n / alpha) - var_n))
n_max = int(np.ceil(avg_n + var_n))
dn = int(np.ceil(4 * (np.sqrt(1+alpha)/alpha) * np.sqrt(var_n)))

nstars = np.arange(n_min, n_max+dn, dn)

print(" kappa: {:.4f}".format(k))
print(" n_min: {:d}  n_max: {:d}".format(n_min, n_max))
print(" dn: {:d}".format(dn))
print(" n_windows: {:d}".format(nstars.size))
