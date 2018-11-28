import matplotlib as mpl
from matplotlib import rc 
import os, glob

homedir = os.environ['HOME']

mpl.rcParams.update({'axes.labelsize': 20})
mpl.rcParams.update({'xtick.labelsize': 20})
mpl.rcParams.update({'ytick.labelsize': 15})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':10})

sys_names = ['2b97', '1msb', '2tsc', '1ycr_mdm2', 'ubiq_merge', '1bi4', '1bmd', '1brs_bn']

name_lut = {
    '2b97': 'Hydrophobin II',
    '2tsc': 'Thymidylate\nsynthase',
    '1bi4': 'HIV\nInteg',
    '1ycr_mdm2': 'MDM2',
    '1bmd': 'Malate\ndehydrogenase',
    '1msb': 'Mannose-\nbinding\nprotein',
    'ubiq_merge': 'Ubiquitin',
    '1brs_bn': 'Barnase'
}
from constants import k

beta = 1/(k * 300)


best_phi = []
best_f1 = []
best_fh = []
best_tpr = []
best_fpr = []

for i, dirname in enumerate(sys_names):
    path = '{}/pred/performance.dat'.format(dirname)

    dat = np.loadtxt(path)

    phi, tp, fp, tn, fn, tpr, fpr, ppv, f_h, f_1, mcc = np.split(dat, 11, 1)

    best_perf = np.argmax(f_h)
    print('{}'.format(name_lut[dirname]))
    print('  phi: {}'.format(phi[best_perf]))

    best_phi.append(beta*phi[best_perf])
    best_f1.append(f_1[best_perf])
    best_fh.append(f_h[best_perf])
    best_tpr.append(tpr[best_perf])
    best_fpr.append(fpr[best_perf])

indices = np.arange(len(sys_names))

labels = [name_lut[name] for name in sys_names]
### bar chart of best phis ###
fig, ax = plt.subplots(figsize=(15,6))
ax.bar(indices, best_phi, width=0.8, color='k')
ax.set_xticks(indices)
ax.set_xticklabels(labels)
ax.set_ylabel(r'$\beta \phi$')
fig.tight_layout()
fig.savefig('{}/Desktop/perf_phi.pdf'.format(homedir), transparent=True)

plt.close('all')

### best fh's ###
fig, ax = plt.subplots(figsize=(15,6))
ax.bar(indices, best_fh, width=0.8, color='k')
ax.set_xticks(indices)
ax.set_xticklabels(labels)
ax.set_ylabel(r'$f_h$')
fig.tight_layout()
fig.savefig('{}/Desktop/perf_fh.pdf'.format(homedir), transparent=True)

plt.close('all')
### best f1's ###
fig, ax = plt.subplots(figsize=(15,6))
ax.bar(indices, best_fh, width=0.8, color='k')
ax.set_xticks(indices)
ax.set_xticklabels(labels)
ax.set_ylabel(r'$f_1$')
fig.tight_layout()
fig.savefig('{}/Desktop/perf_f1.pdf'.format(homedir), transparent=True)

plt.close('all')


### roc points ###
fig, ax = plt.subplots(figsize=(6,5))
ax.plot(best_fpr, best_tpr, 'o')
ax.legend(labels)
