from rhoutils import rho, rho2, phi_1d, gaus, gaustest, interp1d

import time

import numpy as np


def time_it(fn, *args):
    start_time = time.time()
    fn(*args)
    end_time = time.time()

    print("  Total time: {:0.4f}".format(end_time-start_time))


n_pts = 10000000
sigma = 2.4
cutoff = 7.0
max_comp = np.sqrt(3*cutoff**2)
dist_vectors = np.random.uniform(low=-max_comp, high=max_comp, size=(n_pts, 3)).astype(np.float32)

xvals = np.random.uniform(low=-max_comp, high=max_comp, size=(n_pts,)).astype(np.float32)

print("Gaussian function")
time_it(gaus, xvals, sigma, sigma**2)

print("fast Gaussian")
g = gaustest(sigma)
time_it(g, xvals)

newxvals = np.linspace(-max_comp, max_comp, 101, dtype=np.float32)
g_table = gaus(newxvals, sigma, sigma**2)
fast_gaus = interp1d(newxvals, g_table)
print('look up')
time_it(fast_gaus, xvals)

print("phi_1d")
time_it(phi_1d, xvals, sigma, sigma**2, cutoff, cutoff**2)

phi_table = phi_1d(newxvals, sigma, sigma**2, cutoff, cutoff**2)
fast_phi = interp1d(newxvals, phi_table)
print("fast phi")
time_it(fast_phi, xvals)
