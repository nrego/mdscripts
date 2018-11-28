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
header = 'pdb   beta*phi_best   tp    fp    tn    fn   f1   is_heterodimer'


for i,fname in enumerate(fnames):
    

    dat = np.loadtxt(fname)

    tp = dat[:,1]
    fp = dat[:,2]
    tn = dat[:,3]
    fn = dat[:,4]

    tpr = tp/(tp+fn)
    fpr = fp/(tn+fp)

    prec = tp/(tp+fp)


    f1 = 2/((1/tpr) + (1/prec))
    best_perf = np.nanargmax(f1)

    name = os.path.dirname(os.path.dirname(fname))
    sys_names.append(name)
    perf = np.append(perf, f1[best_perf])
    tprs = np.append(tprs, tpr[best_perf])
    precs = np.append(precs, prec[best_perf])
    phis = np.append(phis, beta*dat[best_perf,0])

    this_phi = beta * dat[best_perf, 0]
    outdat[i,0] = name
    outdat[i,1] = this_phi
    outdat[i,2:6] = tp[best_perf],fp[best_perf],tn[best_perf],fn[best_perf]

    is_heter0 = name in heterodimers
    outdat[i,6] = f1[best_perf]
    outdat[i,7] = is_heter0

np.savetxt('all_results.dat', outdat, fmt='%s  %1.2f %1.2f %1.2f %1.2f %1.2f %1.2f %1d', header=header)