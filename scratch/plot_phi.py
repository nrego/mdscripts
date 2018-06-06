from __future__ import division, print_function

import os, glob
import scipy


fnames = sorted( glob.glob('*/n_v_phi.dat') )
fnames = fnames[:2]

radii = []
peak_sus = []


#alpha = lambda dat: 1/dat[0,2]
#c = lambda dat: 1 - (dat[0,1] / dat[0,2])

#alpha = lambda dat: 1/dat[0,1]
alpha = lambda dat: 1
c = lambda dat: 0

avg_x = lambda dat: alpha(dat) * dat[:,1] + c(dat)
var_x = lambda dat: alpha(dat)**2 * dat[:,2]

for i, fname in enumerate(fnames):
    rad = float(os.path.dirname(fname).split('_')[-1]) / 10 # in angstroms
    #rad = labels[i]
    radii.append(rad)
    dirname = os.path.dirname(fname)
    err_fname = os.path.join(dirname, 'err_n_v_phi.dat')
    dat = np.loadtxt(fname)
    err_dat = np.loadtxt(err_fname)

    plt.errorbar(dat[:,0], var_x(dat), yerr=err_dat[:,1], label=r'$R_V={}$'.format(rad))
    #plt.plot(dat[:,0], avg_x(dat), linewidth=6, label=r'$R_V={:.1f}$'.format(rad))
    max_idx = np.argmax(dat[:,2])
    max_phi = dat[max_idx,0]
    peak_sus.append(max_phi)

radii = np.array(radii)
peak_sus = np.array(peak_sus)

#plt.ylabel(r'$\langle X_V \rangle_{\phi}$')
plt.ylabel(r'$\langle \delta {N}^{2}_V \rangle_{\phi}$')
plt.ylabel(r'$\langle X_V \rangle_{\phi}$')
plt.xlabel(r'$\phi$ (kJ/mol)')
plt.legend(loc=1, prop={'size':20})
plt.show()

r_red = 1/radii
expt_slope = 40

slope, inter, r, p, std = scipy.stats.linregress((1/radii), peak_sus)

xvals = np.arange(0, 0.2, 0.001)

# get best fit line with expected slope
slope_diff = peak_sus - r_red*expt_slope
expt_inter = slope_diff.mean()

line_fit = slope*xvals + inter
line_expt = expt_slope*xvals + expt_inter

#plt.plot(xvals, line_fit, 'k-', label='fit', linewidth=6)
#plt.plot(xvals, line_expt, 'k--', label='predicted', linewidth=4)
#plt.plot(r_red, peak_sus, 'o', markersize=12)
#plt.xlabel(r'$\frac{1}{R_V} \; \; (\AA^{-1})$')
#plt.ylabel(r'$\phi^*$ (kJ/mol)')
#plt.legend()
