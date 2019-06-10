import os, glob

fnames = sorted(glob.glob('nstar*'))

nstars = []

f_k = np.loadtxt('f_k_all.dat')
avg_ns = []

for fname in fnames:
    ds = dr.loadPhi('{}/phiout.dat'.format(fname))
    avg_n = ds.data[1000:]['N'].mean()
    avg_ns.append(avg_n)
    is_neg = fname.split('_')[1] == 'neg'
    num = float(fname.split('_')[-1]) 
    num = -num if is_neg else num

    nstars.append(num)

nstars = np.array(nstars)
avg_ns = np.array(avg_ns)
f_k -= f_k.min()

sort_idx = np.argsort(nstars)

plt.plot(nstars[sort_idx], f_k[sort_idx]/50, '-o', )
plt.plot(nstars[sort_idx], avg_ns[sort_idx], '-o')