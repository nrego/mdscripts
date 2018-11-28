## Gather all tp,fp,tn,fn for all systems
#  calculate f1, etc

import matplotlib as mpl
from matplotlib import rc 
import os, glob

homedir = os.environ['HOME']

mpl.rcParams.update({'axes.labelsize': 20})
mpl.rcParams.update({'xtick.labelsize': 15})
mpl.rcParams.update({'ytick.labelsize': 15})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':10})

from constants import k

fnames = glob.glob('*/pred/performance.dat')
beta = 1/(k * 300)

sys_names = []
perf = np.array([])
tprs = np.array([])
precs = np.array([])
phis = np.array([])

heterodimers = ['1brs_bn', '1brs_bs','1bxl', '1tue', '1ycr_mdm2', '1z92_il2', '1z92_il2r', 'ubiq_merge']

outdat = np.zeros((len(fnames), 8), dtype=object)

rc('text', usetex=False)
names = []
phi_f1 = []
phi_fh = []
for i,fname in enumerate(fnames):
    

    dat = np.loadtxt(fname)

    phi, tp, fp, tn, fn, tpr, fpr, ppv, f_h, f_1, mcc = np.split(dat, 11, 1)


    best_f1 = np.argmax(f_1)
    best_fh = np.argmax(f_h)

    name = fname.split('/')[0]
    names.append(name)
    phi_f1.append(phi[best_f1])
    phi_fh.append(phi[best_fh])

indices = np.arange(len(names))

fig, ax = plt.subplots()

ax.bar(indices, phi_f1, label=r'$f_1$', width=0.3)
ax.bar(indices+0.3, phi_fh, label=r'$f_h$', width=0.3)
ax.legend()
ax.set_xticks(indices+0.3)
ax.set_xticklabels(names, rotation='vertical')
plt.show()