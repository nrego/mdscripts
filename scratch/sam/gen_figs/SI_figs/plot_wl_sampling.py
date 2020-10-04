
from scratch.sam.util import *
import matplotlib as mpl

mpl.rcParams.update({'axes.labelsize': 45})
mpl.rcParams.update({'xtick.labelsize': 50})
mpl.rcParams.update({'ytick.labelsize': 50})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':14})

def get_rms(m_mask, positions, prec=2):
    pos = positions[m_mask]
    #if pos.size == 0:
    #    return 0.0
    centroid = pos.mean(axis=0)
    diff = pos-centroid
    sq_dev = (diff**2).sum(axis=1)

    return np.round(np.sqrt(sq_dev.mean()), prec)

# Return true of fname is a k
def get_k(fname, N=36):
    splits = fname.split('/')[1]

    isk, num = splits.split('_')
    isk = isk == 'k' 
    num = float(num)

    if isk:
        num = N-num
    if isk and num == N:
        isk = False

    return isk, num


#ds = np.load('/Users/nickrego/simulations/sam_dos_rms/rmsd_dos_p_06_q_06_kc_012.npz')
#bins = ds['bins'][0]
#bins = np.arange(0, 2.4, 0.1)
bins = np.arange(0, 1.15, 0.05)
bc = bins + 0.5*np.diff(bins)[0]

## FROM sam_data
##  Plot distribution of kc, ko samples to show extent of sampling

ds = np.load("data/sam_pattern_04_04.npz")

N = 16

k_bins = np.arange(N+1)

bb, kk = np.meshgrid(np.append(bins, bins[-1]+np.diff(bins)[0]), np.append(k_bins, N+1))

# d_C, kc
ener_dist_kc = np.zeros((bc.size, k_bins.size))
ener_dist_ko = np.zeros((bc.size, k_bins.size))

ener_dist_kc[:] = np.nan
ener_dist_ko[:] = np.nan

fnames = ds['fnames']
states = ds['states']
energies = ds['energies']

max_rms = 0 

min_energy = np.inf
max_energy = -np.inf

for i, fname in enumerate(fnames):
    is_k, ko = get_k(fname, N=N)
    state = states[i]

    if ko == 36:
        print(fname)
    kc = N - ko

    if is_k:
        #print(fname)
        rms = get_rms(state.methyl_mask, state.positions)
    else:
        rms = get_rms(~state.methyl_mask, state.positions)

    if rms > max_rms:
        max_rms = rms 

    kc_idx = np.digitize(kc, k_bins) - 1
    ko_idx = np.digitize(ko, k_bins) - 1


    bin_idx = np.digitize(rms, bins) - 1


    if is_k:
        if ko_idx == N:
            print(fname)
        if not np.isnan(ener_dist_kc[bin_idx, ko_idx]):
            break
            print(fname)
        ener_dist_kc[bin_idx, ko_idx] = energies[i]
    else:
        #if ko_idx == 0:
        #    print(fname)
        ener_dist_ko[bin_idx, ko_idx] = energies[i]

    if energies[i] > max_energy:
        max_energy = energies[i]
    if energies[i] < min_energy:
        min_energy = energies[i]

norm = plt.Normalize(np.floor(min_energy), np.ceil(max_energy))

plt.close('all')
plt.pcolormesh(bb, kk, ener_dist_ko.T, norm=norm)

ax = plt.gca()
#ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
#ax.set_yticks([0, 6, 12, 18, 24, 30, 36])
ax.set_yticks([0, 4, 8, 12, 16])

mask_kc = ~np.ma.masked_invalid(ener_dist_kc).mask
mask_ko = ~np.ma.masked_invalid(ener_dist_ko).mask
