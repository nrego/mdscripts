import numpy as np
from matplotlib import pyplot as plt
# an array with all your 4901 (possibly autocorrelated) datapoints
# shape: (4901,)
mydat = np.loadtxt('time_samples_lambdav_6_sphere_0.5.out')[:,1]

# bin bounds
bb = np.linspace(0,30,51)
# bin centers
#   shape: (50,)
bc = np.diff(bb)/2.0 + bb[:-1]

# 4901
n_sample = mydat.size
block_size = 200
# 24 blocks
n_blocks = n_sample // block_size
# length of a single bootstrap sample
n_boot_sample = n_blocks * block_size

# indices from which we can take a full block_size-length sample 
#   from mydat
avail_start_indices = n_sample - block_size + 1



# n_bootstrap iterations
n_iter = 1000

# Final array that will hold each bootstrap result -
#    in this case, a single bootstrap result is 
#    the negative log of each histogram bin
boot_results = np.zeros((n_iter, bc.size))

# This array will be filled with a bootstrapped subsample of the original dataset (mydat)
#    each bootstrap iteration
# shape (n_boot_sample, )
this_boot_sample = np.zeros(n_boot_sample)

# for each bootstrap sample...
for i_boot_iter in range(n_iter):
	# a list of start indices from which 
	# to grab each block from the original dataset
	# There are num_blocks of them
	this_boot_start_indices = np.random.randint(avail_start_indices, size=num_blocks)

	# Grab a bootstrap sample from the original dataset (into this_boot_sample)
	for k, boot_start_idx in enumerate(this_boot_start_indices):
		start_idx = k*block_size
		this_boot_sample[start_idx:start_idx+block_size] = mydat[boot_start_idx:boot_start_idx+block_size]


	# no run final analysis on our bootstrap subsample (in this case, find the neg log of the hist bins)

	# Construct a (normalized) histogram from this subsampled dataset
	this_hist, blah = np.histogram(this_boot_sample, bins=bb, normed=True)

	# Record the results for this bootstrap iteration
	boot_results[i_boot_iter, ...] = -np.log(this_hist)


# Get standard errors

# This will find the std deviation across each column of boot_results (i.e. the std
#    dev of each bin's -log(weight) over all n_iter bootstrap samples)
std_errs = boot_results.std(axis=0)

# Use all the data to report the final results
hist_actual = np.histogram(mydat, bins=bb, normed=True)

plt.plot(bc, -np.log(hist_actual), yerr=std_errs)
