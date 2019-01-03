from __future__ import division, print_function

import os, glob
import scipy.stats

from constants import k
from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({'axes.labelsize': 100})
mpl.rcParams.update({'xtick.labelsize': 80})
mpl.rcParams.update({'ytick.labelsize': 80})
mpl.rcParams.update({'axes.titlesize': 50})
mpl.rcParams.update({'legend.fontsize':40})

homedir = os.environ['HOME']

# Some shennanigans to import when running from IPython terminal
try:
    from utils import plt_errorbars
except:
    import imp
    utils = imp.load_source('utils', '{}/mdscripts/scratch/prot_hydrophobicity_comm_scripts/utils.py'.format(homedir))
    plt_errorbars = utils.plt_errorbars


radii = []
peak_sus = []
peak_sus_avg = []
peak_sus_err = []


# Plot 
fig, ax = plt.subplots(figsize=(11, 9))

avg_fnames = list(reversed(sorted( glob.glob('*/n_v_phi.dat') )))
var_fnames = list(reversed(sorted( glob.glob('*/var_n_v_phi.dat') )))
for i, avg_fname in enumerate(avg_fnames):

    if i in [0,1,2,3,4]:
        var_fname = var_fnames[i]
        rad = float(os.path.dirname(avg_fname).split('_')[-1]) / 100
        radii.append(rad)
        dirname = os.path.dirname(avg_fname)

        avg_dat = np.loadtxt(avg_fname)
        # <N_v>0
        avg_0 = avg_dat[0,1]
        var_dat = np.loadtxt(var_fname)
        sus_dat = np.loadtxt('{}/peak_sus.dat'.format(dirname))

        max_idx = np.argmax(var_dat[:,1])
        max_phi = var_dat[max_idx,0]
        peak_sus.append(max_phi)
        
        peak_sus_avg.append(sus_dat[0])
        peak_sus_err.append(sus_dat[1])

        if i in [1,2,3]:
            ax.scatter(var_dat[max_idx, 0], var_dat[max_idx, 1], s=500, linewidths=4, facecolor='none', edgecolor='k', zorder=3)
            ax.plot(var_dat[:,0], var_dat[:,1], linewidth=4, label=r'$R_v={:.1f}$  nm'.format(rad))
            plt_errorbars(var_dat[:,0], var_dat[:,1], var_dat[:,2])
        

radii = np.array(radii)
peak_sus = np.array(peak_sus)
peak_sus_avg = np.array(peak_sus_avg)
peak_sus_err = np.array(peak_sus_err)

#plt.ylabel(r'$\langle X_v \rangle_{\phi}$')
ax.set_ylabel(r'$\chi_v$')
ax.set_xlabel(r'$\beta \phi$')
#ax.legend(handlelength=1)
ax.set_xlim(1.12,2.5)
ax.set_xticks([1.5,2.0])
ax.set_yticks([0,1000, 2000, 3000, 4000,5000])
ymin, ymax = plt.ylim()
ax.set_ylim(0, ymax)
fig.tight_layout()
fig.savefig('{}/Desktop/avg_by_r.pdf'.format(homedir), transparent=True)
fig.show()


## Plot beta phi* v 1/R_v ##
fig, ax = plt.subplots(figsize=(10.30,9.8))
radii_recip = 1/radii

xvals = np.arange(0, 2, 0.01)

# get best fit line thru origin
slope_fit = np.sum(radii_recip*peak_sus)/np.sum(radii_recip*radii_recip)
line_fit = slope_fit*xvals

ax.plot(xvals, line_fit, 'k--', label='fit', linewidth=4, zorder=0)
ax.errorbar(radii_recip, peak_sus_avg, yerr=peak_sus_err, fmt='o', color='#9d0903ff', markersize=20, ecolor='k', 
            elinewidth=1, capsize=10, capthick=1, barsabove=True)
ax.set_xlabel(r'$\frac{1}{R_v} \; \; (\rm{nm}^{-1})$', labelpad=20)
ax.set_ylabel(r'$\beta \phi^*$')
ax.set_xlim(0.9,1.7)
ax.set_ylim(1.12, 2.6)
fig.tight_layout()
fig.savefig('{}/Desktop/phistar_by_r.pdf'.format(homedir), transparent=True)
fig.show()

