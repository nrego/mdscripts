import matplotlib as mpl
from matplotlib import rc 
import os, glob

homedir = os.environ['HOME']

mpl.rcParams.update({'axes.labelsize': 20})
mpl.rcParams.update({'xtick.labelsize': 15})
mpl.rcParams.update({'ytick.labelsize': 15})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':10})

sys_names = ['2b97', '2tsc', '1bi4', '1ycr_mdm2', '1bmd']

name_lut = {
    '2b97': 'hfb II',
    '2tsc': 'thy synth',
    '1bi4': 'HIV Integ',
    '1ycr_mdm2': 'MDM2',
    '1bmd': 'malate dehyd'
}
from constants import k

beta = 1/(k * 300)
fig, ax = plt.subplots(figsize=(8,5))
for i, dirname in enumerate(sys_names):
    path = '{}/pred/performance.dat'.format(dirname)

    dat = np.loadtxt(path)

    tp = dat[:,1]
    fp = dat[:,2]
    tn = dat[:,3]
    fn = dat[:,4]

    tpr = tp/(tp+fn)
    fpr = fp/(tn+fp)

    prec = tp/(tp+fp)

    f1 = 2/((1/tpr) + (1/prec))
    best_perf = np.argmax(f1)


    print('{}'.format(name_lut[dirname]))
    print('  phi: {}'.format(dat[best_perf,0]))

    

    #ax.plot(tpr[best_perf], prec[best_perf], 'o', label=name_lut[dirname])
    ax.bar(i, beta*dat[best_perf,0], width=0.8, color='k')

ax.set_ylabel(r'$\beta \phi$')
ax.set_xticks(np.arange(len(sys_names)))
ax.set_xticklabels([name_lut[sys] for sys in sys_names])

fig.savefig('{}/Desktop/perf_phi.png'.format(homedir), transparent=True)
#ax.legend()
#ax.set_xlim(0,1)
#ax.set_ylim(0,1)

#ax.set_xticks([0,0.5,1])
#ax.set_yticks([0,0.5,1])

#ax.set_xlabel('recall')
#ax.set_ylabel('precision')
#fig.savefig('{}/Desktop/perf_2.pdf'.format(homedir), transparent=True)


