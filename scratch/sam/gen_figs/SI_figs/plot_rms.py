from scratch.sam.util import *
import matplotlib as mpl

mpl.rcParams.update({'axes.labelsize': 45})
mpl.rcParams.update({'xtick.labelsize': 50})
mpl.rcParams.update({'ytick.labelsize': 50})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':14})

def get_rms(m_mask, positions, prec=2):
    pos = positions[m_mask]
    if pos.size == 0:
        return 0.0
    centroid = pos.mean(axis=0)
    diff = pos-centroid
    sq_dev = (diff**2).sum(axis=1)

    return np.round(np.sqrt(sq_dev.mean()), prec)


pt = np.loadtxt("/Users/nickrego/simulations/pattern_sample/k_12/d_075/trial_0/this_pt.dat", dtype=int)
pt_0 = np.array([np.sqrt(3)/2, -6.5])
state = State(pt)


plt.close('all')


state.plot()


blah = (state.positions*2 + pt_0)[state.methyl_mask]
r = get_rms(state.methyl_mask, state.positions*2)
cent = blah.mean(axis=0)

plt.plot(blah[:,0], blah[:,1], 'ko', markersize=12)
plt.plot(cent[0], cent[1], 'ro', markersize=24)

ax = plt.gca()

circ = plt.Circle(cent, radius=r, edgecolor='gray', linestyle='--', linewidth=8, fill=False)
ax.add_artist(circ)

#### Now, plot rms d.o.s.
plt.close('all')
from scipy.special import binom

ds = np.load('/Users/nickrego/simulations/sam_dos_rms/rmsd_dos_p_06_q_06_kc_001.npz')
bins = ds['bins'][0]
bc = bins + 0.5*np.diff(bins)[0]
density = ds['density']
density /= density.sum()
density *= binom(36, 12)
loghist = np.log(density)

#norm = np.log(density.sum())
#loghist -= norm

plt.plot(bc, loghist, '-o', linewidth=8, markersize=24)
ax = plt.gca()

ax.set_xticks([0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
ax.set_yticks([5, 10, 15, 20])

pt_lo = np.loadtxt("/Users/nickrego/simulations/pattern_sample/k_12/d_065/trial_0/this_pt.dat", dtype=int) 
pt_mid = np.loadtxt("/Users/nickrego/simulations/pattern_sample/k_12/d_110/trial_0/this_pt.dat", dtype=int)
pt_hi = np.loadtxt("/Users/nickrego/simulations/pattern_sample/k_12/d_145/trial_0/this_pt.dat", dtype=int)

state_lo = State(pt_lo)
state_mid = State(pt_mid)
state_hi = State(pt_hi)

plt.close('all')
state_lo.plot()
plt.savefig("/Users/nickrego/Desktop/state_lo.png", transparent=True)

plt.close('all')
state_mid.plot()
plt.savefig("/Users/nickrego/Desktop/state_mid.png", transparent=True)

plt.close('all')
state_hi.plot()
plt.savefig("/Users/nickrego/Desktop/state_hi.png", transparent=True)




