import matplotlib as mpl
from matplotlib import rc 
import os, glob

homedir = os.environ['HOME']

mpl.rcParams.update({'axes.labelsize': 40})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':10})

from constants import k

beta = 1/(k * 300)
beta_phi_vals, avg_N, err_avg_N, chi, err_chi = [arr.squeeze() for arr in np.split(np.loadtxt('../prod/phi_sims/Nvphi.dat'), 5, 1)]
## Figure 4 d ##  
## Plot roc curve and susceptiblity w/ f_h

phi, tp, fp, tn, fn, tpr, fpr, prec, f_h, f_1, mcc = [arr.squeeze() for arr in np.split(np.loadtxt('performance.dat'), 11, 1)]

best_idx = np.argmax(f_h)
### phi_opt (best d_h) ###
fig, ax = plt.subplots(figsize=(6,5.5))
ax.plot(fpr, tpr, 'ko', markersize=8)
ax.plot(fpr[best_idx], tpr[best_idx], 'bo', markersize=12)
#ax.plot(fpr[indices], tpr[indices], 'bo', markersize=12)
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_xticks([0,0.5,1])
ax.set_yticks([0,0.5,1])
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
fig.tight_layout()
fig.savefig('{}/Desktop/roc.pdf'.format(homedir), transparent=True)
plt.close('all')


fig, ax1 = plt.subplots(figsize=(7,5))
ax2 = ax1.twinx()
ax1.errorbar(beta_phi_vals, chi, yerr=err_chi, fmt='r-', linewidth=4)
ax1.set_ylabel(r'$\chi_v$', color='r')
ax1.set_xlabel(r'$\beta \phi$')
ax1.tick_params(axis='y', labelcolor='r')

ax2.plot(beta*phi, f_h, 'b-', linewidth=4)
ax2.set_ylabel(r'$d_h$', color='b')
ax2.tick_params(axis='y', labelcolor='b')

fig.tight_layout()
fig.savefig('{}/Desktop/sus_dh_comp.pdf'.format(homedir), transparent=True)
