
import matplotlib as mpl
from matplotlib import rc 
from matplotlib import pyplot as plt
import os, glob
import numpy as np
from constants import k

from IPython import embed

import scipy


################# PRELIM STUFF ############################
COLOR_CONTACT = '#7F00FF' # purple
COLOR_NO_CONTACT = COLOR_NOT_PRED = COLOR_TN = '#7F7F7F' # gray
COLOR_PO = '#0560AD' # blue2
COLOR_NP = '#D7D7D7' # light gray
COLOR_PRED = '#FF7F00' # orange 
COLOR_TP = '#FF007F' # pink
COLOR_FP = '#7F3F00' # dark orange
COLOR_FN = '#3F007F' # dark purple
COLOR_CONVEX = '#FF0000' # red
COLOR_CONCAVE = '#0000FF' # blue

homedir = os.environ['HOME']
savedir = '{}/Desktop'.format(homedir)

mpl.rcParams.update({'axes.labelsize': 40})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':10})
mpl.rcParams.update({'font.size': 20})


################################ INPUT DATA ########################################
beta = 1 /(300*k)

beta_phi_vals_all, tp_np, tp_po, fp_np, fp_po, tn_np, tn_po, fn_np, fn_po = [arr.squeeze() for arr in np.split(np.loadtxt('perf_by_chemistry.dat'), 9, 1)]
beta_phi_vals, avg_N, err_avg_N, smooth_chi, err_smooth_chi, chi, err_chi = [arr.squeeze() for arr in np.split(np.loadtxt('../phi_sims/NvPhi.dat'), 7, 1)]


############### FIND bphi minus, star, plus ##############
# Find phi_minus, phi_star, phi_plus
chi_max_idx = np.argmax(smooth_chi)
chi_max = np.max(smooth_chi)
chi_thresh_mask = smooth_chi < (0.5*chi_max)
chi_minus_idx = np.max(np.where(chi_thresh_mask[:chi_max_idx])) 
chi_plus_idx = np.min(np.where(chi_thresh_mask[chi_max_idx:])) + chi_max_idx 
beta_phi_star = beta_phi_vals[chi_max_idx]
beta_phi_minus = beta_phi_vals[chi_minus_idx]
beta_phi_plus = beta_phi_vals[chi_plus_idx]
########################################################

########### DATA ANALYSIS ############# 
# polar or non-polar dewetted atoms
dewet_po = (tp_po + fp_po).astype(int)
dewet_np = (tp_np + fp_np).astype(int)
# polar or non-polar wet atoms
wet_po = (fn_po + tn_po).astype(int)
wet_np = (fn_np + tn_np).astype(int)

# Should be constants
tot_po = np.unique(dewet_po + wet_po).item()
tot_np = np.unique(dewet_np + wet_np).item()
n_surf = tot_po + tot_np

# Number of wetted or dewetted atoms
dewet_tot = dewet_po + dewet_np
wet_tot = wet_po + wet_np
# Sanity
assert np.abs(np.diff(dewet_tot)).max() <= 1

max_dewet = dewet_tot[-1]
assert max_dewet == dewet_tot.max()
assert np.unique(dewet_tot+wet_tot).item() == n_surf

# Fraction of dewetted surface atoms
frac_dewet = dewet_tot/n_surf
max_x = np.ceil(100*(frac_dewet.max()))/100

# Fraction dewetted atoms that are non-polar
frac_np_dewet = dewet_np / dewet_tot
# Fraction of wet atoms that are polar, versus fraction of surf atoms that are dewetted
frac_po_wet = wet_po / wet_tot

## Find dewetting cohorts (first N that dewet, second N that dewet, etc)
n_cohort = 4
cohort_size = int(np.round(max_dewet / n_cohort))
assert np.round(max_dewet/cohort_size) == n_cohort

print('Cohort size: {} ({} grps)\n'.format(cohort_size, n_cohort))

lb = prev_dewet_tot = 0
prev_dewet_np = 0
prev_dewet_po = 0

cohort_dewet_np = np.zeros(n_cohort)
cohort_dewet_po = np.zeros(n_cohort)
cohort_dewet_tot = np.zeros(n_cohort)
cohort_indices = np.zeros(n_cohort, dtype=int)

for i_cohort in range(n_cohort):

    idx = np.argmin((dewet_tot - (lb+cohort_size))**2)
    cohort_indices[i_cohort] = idx

    this_dewet_tot = dewet_tot[idx]
    assert this_dewet_tot > prev_dewet_tot

    this_wet_tot = wet_tot[idx]

    this_dewet_np = dewet_np[idx]
    this_dewet_po = dewet_po[idx]

    this_delta_tot = this_dewet_tot - prev_dewet_tot
    this_delta_np = this_dewet_np - prev_dewet_np
    this_delta_po = this_dewet_po - prev_dewet_po
    assert this_delta_tot == this_delta_np + this_delta_po

    cohort_dewet_tot[i_cohort] = this_delta_tot
    cohort_dewet_np[i_cohort] = this_delta_np
    cohort_dewet_po[i_cohort] = this_delta_po

    print("Cohort {:>2d}, lb: {:>4d}  tot_dewet: {:>4d}  delta_dewet: {:>2d}".
          format(i_cohort, lb, this_dewet_tot, this_delta_tot))

    lb = prev_dewet_tot = this_dewet_tot
    prev_dewet_np = this_dewet_np
    prev_dewet_po = this_dewet_po


## Plotting symbols
plot_idx = [chi_minus_idx, chi_max_idx, chi_plus_idx]
plot_sym = ['b<', 'bD', 'b>']


####################### PLOT 1 ################################
## fraction non-polar atoms dewetted vs tot frac dewetted atoms
## And fraction polar-atoms still wet vs tot frac dewetted atoms
plt.close('all')

fig, ax = plt.subplots(figsize=(5.25, 5))

ax.plot(frac_dewet, frac_np_dewet, linewidth=4, color=COLOR_NO_CONTACT)
ax.plot(frac_dewet, frac_po_wet, linewidth=4, color=COLOR_PO)
#ax.plot(frac_dewet[cohort_indices], cohort_dewet_np/cohort_dewet_tot, 'o', zorder=100)

for idx, sym in zip(plot_idx, plot_sym):
    exp_idx = np.abs(beta_phi_vals_all - beta_phi_vals[idx]).argmin()
    if frac_dewet[exp_idx] > 0:
        ax.plot(frac_dewet[exp_idx], frac_np_dewet[exp_idx], sym, markersize=16, zorder=3)
        ax.plot(frac_dewet[exp_idx], frac_po_wet[exp_idx], sym, markersize=16, zorder=3)

ax.set_xlim(0, max_x)
ax.set_xticks(np.arange(0,max_x+0.2, 0.2))

fig.tight_layout()
fig.savefig('{}/perf_by_chemistry.pdf'.format(savedir), transparent=True)
plt.close('all')


####################### PLOT 2 ################################
## fraction non-polar atoms dewetted vs tot frac dewetted atoms
## And fraction polar-atoms still wet vs tot frac dewetted atoms
#    now with marginal dewetting
plt.close('all')

fig, ax = plt.subplots(figsize=(5.25, 5))

ax.plot(frac_dewet, frac_np_dewet, linewidth=4, color=COLOR_NO_CONTACT)
ax.plot(frac_dewet, frac_po_wet, linewidth=4, color=COLOR_PO)
pt = ax.plot(frac_dewet[cohort_indices], cohort_dewet_np/cohort_dewet_tot, 'kx', markersize=20, zorder=100)
pt[0].set_clip_on(False)
for idx, sym in zip(plot_idx, plot_sym):
    exp_idx = np.abs(beta_phi_vals_all - beta_phi_vals[idx]).argmin()
    if frac_dewet[exp_idx] > 0:
        ax.plot(frac_dewet[exp_idx], frac_np_dewet[exp_idx], sym, markersize=16, zorder=3)
        ax.plot(frac_dewet[exp_idx], frac_po_wet[exp_idx], sym, markersize=16, zorder=3)

ax.set_xlim(0, max_x)
ax.set_xticks(np.arange(0,max_x+0.2, 0.2))

fig.tight_layout()
fig.savefig('{}/perf_by_chemistry2.pdf'.format(savedir), transparent=True)
plt.close('all')


####################### PLOT 3 ################################
## fraction of atomic 'cohorts' that dewet (first N atoms, next N atoms, etc)
##   that are np or po (cohorts should be roughly 50/50 for intermediately phobic grps of atoms)
plt.close('all')

fig, ax = plt.subplots(figsize=(5, 4.5))

width = 0.7
#ax.bar(2*np.arange(n_cohort), cohort_dewet_np, width, color=COLOR_NP, edgecolor=COLOR_PRED, linewidth=4)
#ax.bar(2*np.arange(n_cohort), cohort_dewet_po, width, bottom=cohort_dewet_np, color=COLOR_PO, edgecolor=COLOR_PRED, linewidth=4)

ax.bar(2*np.arange(n_cohort), 100*(dewet_np[cohort_indices]/dewet_tot[cohort_indices]), width, color=COLOR_NP, edgecolor=COLOR_PRED, linewidth=4)
ax.bar(2*np.arange(n_cohort), 100*(dewet_po[cohort_indices]/dewet_tot[cohort_indices]), width, bottom=100*(dewet_np[cohort_indices]/dewet_tot[cohort_indices]), color=COLOR_PO, edgecolor=COLOR_PRED, linewidth=4)

ax.bar(2*np.arange(n_cohort)+width+0.1, 100*(wet_np[cohort_indices]/wet_tot[cohort_indices]), width, color=COLOR_NP, edgecolor=COLOR_NOT_PRED, linewidth=4)
ax.bar(2*np.arange(n_cohort)+width+0.1, 100*(wet_po[cohort_indices]/wet_tot[cohort_indices]), width, bottom=100*(wet_np[cohort_indices]/wet_tot[cohort_indices]), color=COLOR_PO, edgecolor=COLOR_NOT_PRED, linewidth=4)

#ax.bar(3*np.arange(n_cohort)+2*width+0.3, 100*(dewet_tot[cohort_indices]/n_surf), width, color=COLOR_PRED, edgecolor='k', linewidth=4)
#ax.bar(3*np.arange(n_cohort)+2*width+0.3, 100*(wet_tot[cohort_indices]/n_surf), width, bottom=100*(dewet_tot[cohort_indices]/n_surf), color=COLOR_NOT_PRED, edgecolor='k', linewidth=4)

# Total fraction of non-polar surface atoms
frac_np = 100*tot_np / n_surf

ax.set_xticks([])
ax.set_xticklabels([])
ax.set_yticks([])
ax.set_ylim(0, 110)
xmin, xmax = ax.get_xlim()
ax.plot([-1.2, xmax], [frac_np, frac_np], 'k--', linewidth=4)
ax.set_xlim(-1.2, xmax)

fig.tight_layout()


for i, idx in enumerate(cohort_indices):
    frac_dewet_np = dewet_np[idx] / dewet_tot[idx]
    frac_dewet_po = dewet_po[idx] / dewet_tot[idx]

    frac_wet_np = wet_np[idx] / wet_tot[idx]
    frac_wet_po = wet_po[idx] / wet_tot[idx]

    frac_dewet = dewet_tot[idx] / n_surf
    frac_wet = wet_tot[idx] / n_surf

    ax.text(2*i-0.15, 100*(frac_dewet_np/2.0), r'{:.0f}\%'.format(100*frac_dewet_np), color='white', rotation=90.)
    ax.text(2*i-0.15, 100*(frac_dewet_np + frac_dewet_po/2.0), r'{:.0f}\%'.format(100*frac_dewet_po), color='white', rotation=90.)

    ax.text(2*i+width+0.1-0.15, 100*(frac_wet_np/2.0), r'{:.0f}\%'.format(100*frac_wet_np), color='white', rotation=90.)
    ax.text(2*i+width+0.1-0.15, 100*(frac_wet_np + frac_wet_po/2.0), r'{:.0f}\%'.format(100*frac_wet_po), color='white', rotation=90.)

    #ax.text(3*i+2*width+0.3-0.15, 100*(frac_dewet/2.0), r'{:.0f}\%'.format(100*frac_dewet), color='white', rotation=90.)
    #ax.text(3*i+2*width+0.3-0.15, 100*(frac_dewet + frac_wet/2.0), r'{:.0f}\%'.format(100*frac_wet), color='white', rotation=90.)


fig.savefig('{}/cohort_chemistry.pdf'.format(savedir), transparent=True)
plt.close('all')

'''
for i, (n_np, n_po) in enumerate(zip(cohort_dewet_np, cohort_dewet_po)):
    frac_np = (n_np / (n_np+n_po)) * 100
    frac_po = (n_po / (n_np+n_po)) * 100

    idx = cohort_indices[i]

    ax.text(2*i-0.25, n_np/2.0, r'{:.0f}\%'.format(frac_np), color='white', rotation=90.)
    ax.text(2*i-0.25, n_np+ n_po/2.0, r'{:.0f}\%'.format(frac_po), color='white', rotation=90.)

    n_np_wet = wet_np[idx]
    n_po_wet = wet_po[idx]
    frac_np_wet = (n_np_wet / (n_np_wet+n_po_wet)) * 100
    frac_po_wet = (n_po_wet / (n_np_wet+n_po_wet)) * 100

    ax.text(2*i-0.25+width, n_np_wet/2.0, r'{:.0f}\%'.format(frac_np_wet), color='white', rotation=90.)
    ax.text(2*i-0.25+width, n_np_wet + n_po_wet/2.0, r'{:.0f}\%'.format(frac_po_wet), color='white', rotation=90.)
'''





print("beta phi_star: {}".format(beta_phi_vals[chi_max_idx]))
print("beta phi_minus: {}".format(beta_phi_minus))
print("beta phi_plus: {}".format(beta_phi_plus))


