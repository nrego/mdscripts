import os, glob

fnames = sorted(glob.glob('density_*'))

bins = None
omega_tot = None

from constants import k

for fname in fnames:
    ds = np.load(fname)
    if bins is None:
        bins = ds['bins']
    else:
        assert np.array_equal(bins, ds['bins'])
    if omega_tot is None:
        omega_tot = ds['omega']
    else:
        omega_tot += ds['omega']

    occ = ds['omega'] > 0
    min_e = bins[:-1][occ].min()
    max_e = bins[:-1][occ].max()
    print('K_C: {:d}'.format(int(ds['k_ch3'])))
    print('  energy range: {:.2f} to {:.2f} ({:.2f})'.format(min_e, max_e, max_e-min_e))

entropy = k*np.log(omega_tot)
mask = ~np.ma.masked_invalid(entropy).mask

coef = np.polyfit(bins[:-1][mask], entropy[mask], deg=5)
p = np.poly1d(coef)

plt.plot(bins[:-1], p(bins[:-1]))
