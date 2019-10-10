import matplotlib as mpl
from matplotlib import rc 
import os, glob

homedir = os.environ['HOME']

mpl.rcParams.update({'axes.labelsize': 50})
mpl.rcParams.update({'xtick.labelsize': 50})
mpl.rcParams.update({'ytick.labelsize': 50})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':20})

sys_names = ['2tsc_pred', '1msb_pred', '1pp2', '1ycr_mdm2', '2z59_ubiq']

from constants import k

beta = 1/(k * 300)

# phi corresponding to maximum d_h
best_phi = []
# peak sus
beta_phi_star = []
best_f1 = []
best_dh = []
best_tpr = []
best_fpr = []
best_tp = []
best_fp = []
best_tn = []
best_fn = []

star_phi = []
star_f1 = []
star_dh = []
star_tpr = []
star_fpr = []
star_tp = []
star_fp = []
star_tn = []
star_fn = []

fig, ax = plt.subplots(figsize=(5.2,5))

for i, dirname in enumerate(sys_names):
    path = '{}/old_prot_all/pred_reweight/performance.dat'.format(dirname)
    
    # need variance for phi*
    # This is in beta*phi...
    ntwid_dat = np.loadtxt('{}/old_prot_all/phi_sims/NvPhi.dat'.format(dirname))
    this_beta_phi_star = ntwid_dat[np.argmax(ntwid_dat[:,3]), 0]
    beta_phi_star.append(this_beta_phi_star)

    perf_dat = np.loadtxt(path)

    bphi, tp, fp, tn, fn, tpr, fpr, ppv, f_1, d_h, mcc = [arr.squeeze() for arr in np.split(perf_dat, 11, 1)]

    best_perf = np.argmax(d_h)
    print('{}'.format(dirname))
    print(' beta phi star: {}'.format(this_beta_phi_star))
    print(' beta phi opt: {}'.format(bphi[best_perf]))
    print(' beta phi best f1: {}'.format(bphi[np.argmax(f_1)]))

    best_phi.append(bphi[best_perf])
    best_f1.append(f_1[best_perf])
    best_dh.append(d_h[best_perf])
    best_tpr.append(tpr[best_perf])
    best_fpr.append(fpr[best_perf])

    best_tp.append(tp[best_perf])
    best_fp.append(fp[best_perf])
    best_tn.append(tn[best_perf])
    best_fn.append(fn[best_perf])

    ## Find points corresponding to phi_star
    star_idx = np.digitize(this_beta_phi_star, bphi) - 1
    star_phi.append(bphi[star_idx])
    star_f1.append(f_1[star_idx])
    star_dh.append(d_h[star_idx])
    star_tpr.append(tpr[star_idx])
    star_fpr.append(fpr[star_idx])

    star_tp.append(tp[star_idx])
    star_fp.append(fp[star_idx])
    star_tn.append(tn[star_idx])
    star_fn.append(fn[star_idx])

    ax.plot(fpr[best_perf], tpr[best_perf], 'o', label=i)

## Put all data together in a nice array ##
best_dat = np.dstack((sys_names,  best_phi, best_tp, best_fp, best_tn, best_fn, best_tpr, best_fpr, best_dh)).squeeze().astype(object)
for i_col in range(1, best_dat.shape[1]):
    best_dat[:,i_col] = best_dat[:,i_col].astype(float)

## Put all data together in a nice array ##
star_dat = np.dstack((sys_names, star_phi, star_tp, star_fp, star_tn, star_fn, star_tpr, star_fpr, star_dh)).squeeze().astype(object)
for i_col in range(1, star_dat.shape[1]):
    star_dat[:,i_col] = star_dat[:,i_col].astype(float)

header = 'name  beta*phi_opt  tp_opt  fp_opt  tn_opt  fn_opt  tpr_opt  fpr_opt  d_h_opt'
fmt_str = '%s  %0.4f  %1d  %1d  %1d  %1d  %0.4f  %0.4f  %0.4f'
np.savetxt('{}/Desktop/protein_prediction_summary_phi_opt.dat'.format(homedir), best_dat, header=header, fmt=fmt_str)

header = 'name  beta*phi_star  tp_star  fp_star  tn_star  fn_star  tpr_star  fpr_star  d_h_star'
fmt_str = '%s  %0.4f  %1d  %1d  %1d  %1d  %0.4f  %0.4f  %0.4f'
np.savetxt('{}/Desktop/protein_prediction_summary_phi_star.dat'.format(homedir), star_dat, header=header, fmt=fmt_str)


## Plot ROC points ###
ax.legend()
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_xticks([0,0.5,1])
ax.set_yticks([0,0.5,1])
fig.tight_layout()
fig.savefig('{}/Desktop/roc_points.pdf'.format(homedir), transparent=True)
plt.close('all')


indices = np.arange(len(sys_names))


### phi_opt (best d_h) ###
fig, ax = plt.subplots(figsize=(8,4))
ax.bar(indices, np.squeeze(best_phi), width=0.8, color='k')
ax.set_xticks(indices)
#ax.set_xticklabels(labels)
ax.set_ylabel(r'$\beta \phi_\mathrm{opt}$')
fig.tight_layout()
fig.savefig('{}/Desktop/perf_phi.pdf'.format(homedir), transparent=True)

plt.close('all')

### best dh's ###
fig, ax = plt.subplots(figsize=(8,4))
ax.bar(indices, np.squeeze(best_dh), width=0.8, color='k')
ax.set_xticks(indices)
#ax.set_xticklabels(labels)
ax.set_ylabel(r'$d_h$')
ax.set_ylim(0,1)
fig.tight_layout()
fig.savefig('{}/Desktop/perf_dh.pdf'.format(homedir), transparent=True)

plt.close('all')
### best f1's ###
fig, ax = plt.subplots(figsize=(10,6))
ax.bar(indices, np.squeeze(best_f1), width=0.8, color='k')
ax.set_xticks(indices)
#ax.set_xticklabels(labels)
ax.set_ylabel(r'$f_1$')
fig.tight_layout()
fig.savefig('{}/Desktop/perf_f1.pdf'.format(homedir), transparent=True)

plt.close('all')

### phistar ###
fig, ax = plt.subplots(figsize=(10,6))
ax.bar(indices, beta_phi_star, width=0.8, color='k')
ax.set_xticks(indices)
#ax.set_xticklabels(labels)
ax.set_ylabel(r'$\beta \phi^*$')
fig.tight_layout()
fig.savefig('{}/Desktop/phi_star.pdf'.format(homedir), transparent=True)

plt.close('all')

### phi_opt/phi_star ###
fig, ax = plt.subplots(figsize=(8,4))
ax.bar(indices, np.squeeze(best_phi)/np.array(beta_phi_star), width=0.8, color='k')
ax.set_xticks(indices)
#ax.set_xticklabels(labels)
ax.set_ylabel(r'$\frac{\phi_\mathrm{opt}}{\phi^*}$')
fig.tight_layout()
fig.savefig('{}/Desktop/phi_ratio.pdf'.format(homedir), transparent=True)

plt.close('all')

fig, ax = plt.subplots(figsize=(6,5.5))
ax.bar(indices, np.squeeze(best_dh), width=0.4, color='k', label=r'$d_\mathrm{h}$')
ax.bar(indices+0.4, np.squeeze(best_phi)/np.array(beta_phi_star), width=0.4, color='b', label=r'$\frac{\phi_\mathrm{opt}}{\phi^*}$')
ax.set_xticks(indices+0.2)
ax.set_xticklabels([])
#ax.set_xticklabels(labels)
ax.set_ylim(0,1.15)
fig.tight_layout()
fig.savefig('{}/Desktop/double_bar.pdf'.format(homedir), transparent=True)

