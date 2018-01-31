
mydat = ds_forw
start = 2000
#dat = np.squeeze(np.array(mydat.dhdl[start:]))
dat = np.squeeze(-np.array(mydat.data[start:][0.1]))

size = dat.size
block_sizes = np.arange(1, size/4, 1)

std_errs = np.zeros_like(block_sizes).astype(float)

for i, block_size in enumerate(block_sizes):
    n_blocks = size / block_size
    rem = size % block_size

    this_dat = dat[rem:].reshape((n_blocks, block_size))

    std_errs[i] = this_dat.mean(axis=1).std(ddof=1)


num_blocks = size / block_sizes

plt.plot(block_sizes, std_errs / np.sqrt(num_blocks - 1))