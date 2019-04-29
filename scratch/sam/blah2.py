import os, glob

fnames = sorted(glob.glob('density_*'))

bins = None
omega_tot = None
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
    print('K_C: {:d}'.format(ds['k_ch3']))
    print('  energy range: {:.2f} to {:.2f} ({:.2f})'.format(min_e, max_e, max_e-min_e))