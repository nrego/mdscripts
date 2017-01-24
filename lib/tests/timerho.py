from rhoutils import rho, rho2

import time

import numpy as np

n_pts = 10000000
sigma = 2.4
cutoff = 7.0
max_comp = np.sqrt(3*cutoff**2)
dist_vectors = np.random.uniform(low=-max_comp, high=max_comp, size=(n_pts, 3)).astype(np.float64)


start_time = time.time()
rho(dist_vectors, sigma, sigma**2, cutoff, cutoff**2)
end_time = time.time()

print "Total time with phi_fast: {}".format(end_time - start_time)

start_time = time.time()
rho2(dist_vectors, sigma, sigma**2, cutoff, cutoff**2)
end_time = time.time()
print "Total time with phi_1d: {}".format(end_time - start_time)