from __future__ import division, print_function

import os, glob
import scipy.stats

from constants import k
from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({'axes.labelsize': 80})
mpl.rcParams.update({'xtick.labelsize': 60})
mpl.rcParams.update({'ytick.labelsize': 60})
mpl.rcParams.update({'axes.titlesize': 50})
mpl.rcParams.update({'legend.fontsize':40})


beta = 1/(k*300)
fnames = sorted( glob.glob('*/var_n_v_phi.dat') )

radii = []
peak_sus = []

def plt_errorbars(bb, vals, errs, **kwargs):
    plt.fill_between(bb, vals-errs, vals+errs, alpha=0.5, **kwargs)

#alpha = lambda dat: 1/dat[0,2]
#c = lambda dat: 1 - (dat[0,1] / dat[0,2])

#alpha = lambda dat: 1/dat[0,1]
alpha = lambda dat: 1
c = lambda dat: 0

avg_x = lambda dat: alpha(dat) * dat[:,1] + c(dat)
var_x = lambda dat: alpha(dat)**2 * dat[:,2]

fig, ax = plt.subplots(figsize=(11.1,8.8))
#for i, fname in enumerate(reversed(fnames[1:-1])):
for i, fname in enumerate(reversed(fnames)):

    rad = float(os.path.dirname(fname).split('_')[-1]) / 100
    radii.append(rad)
    dirname = os.path.dirname(fname)

    dat = np.loadtxt(fname)

    max_idx = np.argmax(dat[:,2])
    max_phi = dat[max_idx,0]
    peak_sus.append(max_phi)

    if i in [1,2,3]:
        #plt.errorbar(dat[:,0], var_x(dat), yerr=err_dat[:,1], label=r'$R_V={}$'.format(rad))
        ax.scatter(dat[max_idx, 0], dat[max_idx, 1], s=500, linewidths=4, facecolor='none', edgecolor='k', zorder=3)
        ax.plot(dat[:,0], dat[:,1], linewidth=8, label=r'$R_v={:.1f}$  nm'.format(rad))
        

radii = np.array(radii)
peak_sus = beta*np.array(peak_sus)

#plt.ylabel(r'$\langle X_V \rangle_{\phi}$')
ax.set_ylabel(r'$\chi_v$')
ax.set_xlabel(r'$\beta \phi$')
ax.legend(loc=1,handlelength=1)
ax.set_xlim(1,2.5)
ax.set_xticks([1.5,2.0])
ymin, ymax = plt.ylim()
ax.set_ylim(0, ymax)
fig.tight_layout()
plt.savefig('/Users/nickrego/Desktop/sus_by_r.pdf')
plt.show()

fig, ax = plt.subplots(figsize=(9.5,8.5))
r_red = 1/radii
expt_slope = 40

slope, inter, r, p, std = scipy.stats.linregress((1/radii), peak_sus)

xvals = np.arange(0, 2, 0.01)

# get best fit line with expected slope
slope_fit = np.sum(r_red*peak_sus)/np.sum(r_red*r_red)
line_fit = slope_fit*xvals

slope_expt = beta*40/10
inter_expt = np.mean(peak_sus - slope_expt*r_red)
line_expt = slope_expt*xvals + inter_expt

ax.plot(xvals, line_fit, 'k:', label='fit', linewidth=16)
#plt.plot(xvals, line_expt, 'k--', label='predicted', linewidth=4)
ax.plot(r_red, peak_sus, 'o', color='#9d0903ff', markersize=30)
ax.set_xlabel(r'$\frac{1}{R_V} \; \; (\rm{nm}^{-1})$', labelpad=20)
ax.set_ylabel(r'$\beta \phi^*$')
ax.set_xlim(0.9,1.7)
ax.set_ylim(0.45, 1.04)
fig.tight_layout()
plt.savefig('/Users/nickrego/Desktop/phistar_by_r.pdf')
plt.show()
#plt.legend()
