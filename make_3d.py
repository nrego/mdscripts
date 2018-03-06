from __future__ import division, print_function

import westpa
from fasthist import histnd, normhistnd
import numpy as np
import matplotlib
mpl = matplotlib
from matplotlib import pyplot as plt
from matplotlib.image import NonUniformImage, imread
#import visvis as vv

from IPython import embed

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

mpl.rcParams.update({'axes.labelsize': 30})
mpl.rcParams.update({'xtick.labelsize': 20})
mpl.rcParams.update({'ytick.labelsize': 20})
mpl.rcParams.update({'axes.titlesize': 50})

def iter_wham(weights, n_tots, bias_mat):

    n_tot = n_tots.sum()
    n_windows = weights.size
    new_weights = np.zeros_like(weights)

    exp = weights - bias_mat
    denom = np.dot(np.exp(exp), n_tots)
    
    for k in range(n_windows):
    # for each observation i...
        #for i in range(n_tot):
        #    num = np.exp(-bias_mat[i,k])
        #    exp = weights - bias_mat[i, :]
        #    denom = n_tots * np.exp(exp)
        #    denom = denom.sum()

        #   new_weights[k] += num / denom
        num = np.exp(-bias_mat[:, k])
        new_weights[k] = -np.log(np.sum(num/denom))

    return new_weights

binbounds = [np.arange(-181,184,4), np.arange(-181,184,4), np.arange(0.0, 80.25, 1)]
phi_binbounds = binbounds[0]
psi_binbounds = binbounds[1]
ntwid_binbounds = binbounds[2]

iter_start = 1000


temp = 300
beta = 1/(temp*8.3144598e-3)
phi_vals = np.array([0, 2.0, 4.0, 5.0, 5.5, 6.0, 8.0, 10.0, 15.0, 20])

free_energies = np.array([  -0, 44.589521, 83.541158, 99.606209, 105.466612, 109.408471,116.394083, 118.821335, 120.544176, 121.066123])

files = ['phi_{:05g}/west.h5'.format(phi*1000) for phi in phi_vals]

# Phi biases - Not to be confused with phi angles of pcoord
phi_vals = beta * phi_vals
n_windows = phi_vals.size

data_managers = []

for filename in files:
    dm = westpa.rc.new_data_manager()
    dm.we_h5filename = filename
    dm.open_backing()

    data_managers.append(dm)

# Total number of samples (n_iter * n_segs_per_iter) for each window
n_tots = np.zeros(phi_vals.size, dtype=int)

all_data = []
all_ntwids = []
all_weights = []

avg_ntwids = []
for i,dm in enumerate(data_managers):
    iter_stop = dm.current_iteration
    n_tot = dm.we_h5file['summary']['n_particles'][iter_start-1:iter_stop-1].sum()

    n_tots[i] = n_tot

    iter_stop = dm.current_iteration
    this_ntwid = 0
    for n_iter in xrange(iter_start, iter_stop):
        iter_group = dm.get_iter_group(n_iter)
        ntwids = iter_group['auxdata/ntwid'][:,-1]
        weights = iter_group['seg_index']['weight']
        this_ntwid += np.dot(ntwids, weights)
        all_data.append(ntwids*weights)
        all_ntwids.append(ntwids)
        all_weights.append(weights)

    avg_ntwids.append(this_ntwid/(iter_stop-iter_start))

all_data = np.concatenate(all_data)
all_ntwids = np.concatenate(all_ntwids)
all_weights = np.concatenate(all_weights)
#bias_mat = np.zeros((all_data.size, n_windows), dtype=np.float32)

#for i, phi in enumerate(phi_vals):
#    bias_mat[:,i] = -np.log(all_weights) + (phi * all_ntwids)

## Now for the meat...

# total histogram
hist = np.zeros((phi_binbounds.size-1, psi_binbounds.size-1, ntwid_binbounds.size-1), dtype=np.float64)

## torsional histogram for each phi value (WE simulation)
phi_hist = []
## For each Phi window
for i, dm in enumerate(data_managers):
    iter_stop = dm.current_iteration
    this_phi_hist = np.zeros((phi_binbounds.size-1, psi_binbounds.size-1, ntwid_binbounds.size-1), dtype=np.float64)
    # For each iteration
    for n_iter in xrange(iter_start, iter_stop):
        iter_group = dm.get_iter_group(n_iter)

        phis = iter_group['pcoord'][:,-1,0]
        psis = iter_group['pcoord'][:,-1,1]
        ntwids = iter_group['auxdata/ntwid'][:,-1]
        nregs = iter_group['auxdata/nreg'][:,-1]
        weights = iter_group['seg_index']['weight']

        assert phis.shape[0] == psis.shape[0] == ntwids.shape[0] == weights.shape[0]

        n_segs = phis.shape[0]

        # stack the phi, psi, and Ntwid (or N) vals for each seg into single array
        vals = np.array([phis,psis,nregs]).T
        assert vals.shape == (n_segs, 3)

        denom = np.dot(phi_vals[:,np.newaxis], ntwids[np.newaxis, :])
        assert denom.shape == (n_windows, n_segs)

        denom -= free_energies[:, np.newaxis]
        denom = np.exp(-denom)
        denom *= n_tots[:, np.newaxis] 
        denom = np.sum(denom, axis=0)

        histnd(vals, binbounds, n_segs*weights/denom, out=hist, binbound_check=False)

        # Just this we run
        histnd(np.array([phis,psis]).T, binbounds[:-1], weights, out=this_phi_hist, binbound_check=False)

    this_phi_hist /= this_phi_hist.sum()
    phi_hist.append(this_phi_hist)

for dm in data_managers:
    dm.close_backing()

# normalize the fucker
normhistnd(hist, binbounds)

phi_binctrs = phi_binbounds[:-1] + np.diff(phi_binbounds)/2.0
psi_binctrs = psi_binbounds[:-1] + np.diff(psi_binbounds)/2.0

vmin, vmax = 0, 12
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
extent = (phi_binbounds[0], phi_binbounds[-1], psi_binbounds[0], psi_binbounds[-1])

min_energy = 20.0
for i in range(64):
    thishist = hist[:,:,i].copy()
    normhistnd(thishist, binbounds[0:2])
    loghist = -np.log(thishist)
    if loghist.min() < min_energy:
        print(i)
        min_energy = loghist.min()

dirnames = np.array([0, 20, 40, 50, 55, 60, 80, 100, 150, 200])
forw_initial = np.loadtxt('phi_sims/forw/rama.xvg', usecols=(0,1), comments=['@','#'])
revr_initial = np.loadtxt('phi_sims/revr/rama.xvg', usecols=(0,1), comments=['@','#'])

for i in range(n_windows):
    phi_val = phi_vals[i] / beta
    plt.figure()
    ax = plt.gca()
    thishist = phi_hist[i]
    loghist = -np.log(thishist)
    loghist -= loghist.min()

    loghist[loghist==float('inf')] = vmax

    im = ax.imshow(loghist.T, extent=extent, interpolation='nearest', origin='lower', alpha=0.75,
                   cmap=cm.nipy_spectral, norm=norm, aspect='auto')
    cont = ax.contour(loghist.T, extent=extent, origin='lower', levels=np.arange(vmin,vmax,1),
                      colors='k', linewidths=1.0)
    cb = plt.colorbar(im)
    ax.set_title('$\phi={:0.1f}$'.format(phi_val), fontsize=30)
    ax.set_xlim(-180,100)
    ax.set_ylabel(r"$\Psi$")
    ax.set_xlabel(r"$\Phi$")
    plt.tight_layout()
    plt.savefig('plot_phi_{:0.1f}.png'.format(phi_val))

    forw_rama = np.loadtxt("phi_sims/forw/phi_{:03d}/rama.xvg".format(dirnames[i]), usecols=(0,1), comments=['@','#'])
    revr_rama = np.loadtxt("phi_sims/revr/phi_{:03d}/rama.xvg".format(dirnames[i]), usecols=(0,1), comments=['@','#'])

    plt.plot(forw_rama[:,0], forw_rama[:,1], 'o', label='Forward', color='r', alpha=0.7)
    plt.plot(revr_rama[:,0], revr_rama[:,1], 'x', label='Reverse', markeredgecolor='k', alpha=0.7)

    plt.plot(forw_initial[0], forw_initial[1], 'o', label='Forward start', color='r', markeredgecolor='k', markersize=18)
    plt.plot(revr_initial[0], revr_initial[1], 'X', label='Reverse start', color='r', markeredgecolor='k', markersize=18)

    plt.legend(loc=8)
    plt.savefig('plot_phi_marker{:0.1f}.png'.format(phi_val))

for i in range(63):
    plt.figure()
    ax = plt.gca()
    thishist = hist[:,:,i].copy()
    normhistnd(thishist, binbounds[0:2])
    loghist = -np.log(thishist)
    #loghist -= min_energy
    loghist -= loghist.min()
    
    loghist[loghist==float('inf')] = vmax

    im = ax.imshow(loghist.T, extent=extent, interpolation='nearest', origin='lower', alpha=0.75,
                   cmap=cm.nipy_spectral, norm=norm, aspect='auto')
    cont = ax.contour(loghist.T, extent=extent, origin='lower', levels=np.arange(vmin,vmax,1),
                      colors='k', linewidths=1.0)
    cb = plt.colorbar(im)
    ax.set_title('$N_V={:02g}$'.format(i), fontsize=30)
    ax.set_xlim(-180,100)
    plt.tight_layout()
    plt.savefig('plot_{:02g}.png'.format(i))

fig = plt.figure()
ax = fig.gca(projection='3d')
PHI, PSI = np.meshgrid(phi_binctrs, psi_binctrs)
z = np.ones_like(PHI)
left_line_xpts = []
left_line_ypts = []
left_line_zpts = []
right_line_xpts = []
right_line_ypts = []
right_line_zpts = []
for i in range(48,61,6):
    myhist = hist[:,:,i].copy()
    normhistnd(myhist, binbounds[0:2])
    loghist = -np.log(myhist)
    loghist -= loghist.min()

    left_min_idx = np.argmin(loghist[0:65, :])
    left_line_xpts.append(phi_binbounds[left_min_idx//181])
    left_line_ypts.append(psi_binbounds[left_min_idx%181])
    left_line_zpts.append(i)

    right_min_idx = np.argmin(loghist[65:, :])
    right_line_xpts.append(phi_binbounds[right_min_idx//181 + 65])
    right_line_ypts.append(psi_binbounds[right_min_idx%181])
    right_line_zpts.append(i)

    surf = ax.contourf(PHI, PSI, loghist.T, 100, zdir='z', offset=i, cmap=cm.jet, vmin=0, vmax=18)
    del myhist, loghist
    ax.plot([-180, 180], [-180, -180], [i, i], color='k')
    ax.plot([-180, 180], [180, 180], [i, i], color='k')
    ax.plot([-180, -180], [-180, 180], [i, i], color='k')
    ax.plot([180, 180], [-180, 180], [i, i], color='k')


#ax.plot(left_line_xpts, left_line_ypts, left_line_zpts)
#ax.plot(right_line_xpts, right_line_ypts, right_line_zpts)
ax.set_zlim(48, 60.1)
ax.set_zlabel('$N_V$')
ax.set_xlabel('$\Phi$')
ax.set_ylabel('$\Psi$')

