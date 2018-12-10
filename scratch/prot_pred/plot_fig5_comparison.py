import matplotlib as mpl
from matplotlib import rc 
import os, glob

homedir = os.environ['HOME']

mpl.rcParams.update({'axes.labelsize': 30})
mpl.rcParams.update({'xtick.labelsize': 15})
mpl.rcParams.update({'ytick.labelsize': 20})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':15})

sys_names = ['2tsc', '1msb', '1pp2', '1ycr_mdm2', 'ubiq_merge', '1bi4', '1brs_bn']

name_lut = {
    '2b97': 'Hydrophobin',
    '2tsc': 'Thymidylate\nsynthase',
    '1bi4': 'HIV\nInteg',
    '1ycr_mdm2': 'MDM2',
    '1bmd': 'Malate\ndehydrogenase',
    '1msb': 'Mannose-\nbinding\nprotein',
    'ubiq_merge': 'Ubiquitin',
    '1brs_bn': 'Barnase',
    '1pp2': 'Phospholipase\nA2'
}
from constants import k

beta = 1/(k * 300)

# phi corresponding to maximum d_h
best_phi = []
# peak sus
phi_star = []
best_f1 = []
best_dh = []
best_tpr = []
best_fpr = []
best_indices = []
best_tp = []
best_fp = []
best_tn = []
best_fn = []

fig, ax = plt.subplots(figsize=(5.2,5))

for i, dirname in enumerate(sys_names):
    path = '{}/pred/performance.dat'.format(dirname)

    # need variance for phi*
    # This is in beta*phi...
    ntwid_dat = np.loadtxt('{}/prod/phi_sims/Nvphi.dat'.format(dirname))
    this_phi_star = ntwid_dat[np.argmax(ntwid_dat[:,3]), 0]
    phi_star.append(this_phi_star)

    # irritatingly, this is kJ/mol
    dat = np.loadtxt(path)

    phi, tp, fp, tn, fn, tpr, fpr, ppv, d_h, f_1, mcc = [arr.squeeze() for arr in np.split(dat, 11, 1)]

    best_perf = np.argmax(d_h)
    best_indices.append(best_perf)
    print('{}'.format(name_lut[dirname]))
    print('  phi: {}'.format(phi[best_perf]))

    best_phi.append(beta*phi[best_perf])
    best_f1.append(f_1[best_perf])
    best_dh.append(d_h[best_perf])
    best_tpr.append(tpr[best_perf])
    best_fpr.append(fpr[best_perf])

    best_tp.append(tp[best_perf])
    best_fp.append(fp[best_perf])
    best_tn.append(tn[best_perf])
    best_fn.append(fn[best_perf])

    ax.plot(fpr[best_perf], tpr[best_perf], 'o', label=name_lut[sys_names[i]])

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

labels = [name_lut[name] for name in sys_names]

### phi_opt (best d_h) ###
fig, ax = plt.subplots(figsize=(8,4))
ax.bar(indices, np.squeeze(best_phi), width=0.8, color='k')
ax.set_xticks(indices)
ax.set_xticklabels(labels)
ax.set_ylabel(r'$\beta \phi_\mathrm{opt}$')
fig.tight_layout()
fig.savefig('{}/Desktop/perf_phi.pdf'.format(homedir), transparent=True)

plt.close('all')

### best dh's ###
fig, ax = plt.subplots(figsize=(8,4))
ax.bar(indices, np.squeeze(best_dh), width=0.8, color='k')
ax.set_xticks(indices)
ax.set_xticklabels(labels)
ax.set_ylabel(r'$d_h$')
ax.set_ylim(0,1)
fig.tight_layout()
fig.savefig('{}/Desktop/perf_dh.pdf'.format(homedir), transparent=True)

plt.close('all')
### best f1's ###
fig, ax = plt.subplots(figsize=(10,6))
ax.bar(indices, np.squeeze(best_f1), width=0.8, color='k')
ax.set_xticks(indices)
ax.set_xticklabels(labels)
ax.set_ylabel(r'$f_1$')
fig.tight_layout()
fig.savefig('{}/Desktop/perf_f1.pdf'.format(homedir), transparent=True)

plt.close('all')

### phistar ###
fig, ax = plt.subplots(figsize=(10,6))
ax.bar(indices, phi_star, width=0.8, color='k')
ax.set_xticks(indices)
ax.set_xticklabels(labels)
ax.set_ylabel(r'$\beta \phi^*$')
fig.tight_layout()
fig.savefig('{}/Desktop/phi_star.pdf'.format(homedir), transparent=True)

plt.close('all')

### phi_opt/phi_star ###
fig, ax = plt.subplots(figsize=(8,4))
ax.bar(indices, np.squeeze(best_phi)/np.array(phi_star), width=0.8, color='k')
ax.set_xticks(indices)
ax.set_xticklabels(labels)
ax.set_ylabel(r'$\frac{\phi_\mathrm{opt}}{\phi^*}$')
fig.tight_layout()
fig.savefig('{}/Desktop/phi_ratio.pdf'.format(homedir), transparent=True)

plt.close('all')


