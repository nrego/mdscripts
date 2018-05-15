import glob, os
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams.update({'axes.labelsize': 40})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize': 40})


fnames = sorted( glob.glob('*/phi_sims/out.dat') )
#fnames = sorted( glob.glob('*/out.dat') )
kt = 8.3144598e-3*300
for fname in fnames:
    dirname = os.path.dirname(fname)
    protname = os.path.dirname(dirname)

    dat = np.loadtxt(fname)

    ntwid_avg_err = np.loadtxt('{}/ntwid_err.dat'.format(dirname))
    ntwid_var_err = np.loadtxt('{}/ntwid_var_err.dat'.format(dirname))

    ## N v Phi
    plt.errorbar(dat[:,0]*kt, dat[:,1], yerr=ntwid_avg_err, fmt='-o', linewidth=6, markersize=12)
    #plt.plot(dat[:,0], dat[:,-1], '-o', linewidth=6, markersize=12, label='{}'.format(dirname))
    plt.xlabel(r'$\phi$ (kJ/mol)')
    plt.xticks( np.arange(-2, 14.0, 2.0) )
    plt.xlim(-0.5, 10.5)

    plt.tight_layout()
    plt.savefig('{}_n_v_phi.png'.format(protname))
    plt.clf()

    ## Var(N) v Phi

    plt.errorbar(dat[:,0]*kt, dat[:,-1], yerr=ntwid_var_err, fmt='-o', linewidth=6, markersize=12)
    plt.xlabel(r'$\phi$ (kJ/mol)')
    plt.xticks( np.arange(-2, 14.0, 2.0) )
    plt.xlim(-0.5, 10.5)
    ylim = plt.ylim()
    ymin,ymax = ylim

    yfloor = np.floor(ymin/100)*100
    yceil = np.ceil(ymax/100)*100 + 200
    nsteps = int(np.floor(((yceil-yfloor)/5)/100) * 100)
    yticks = np.arange(yfloor, yceil, nsteps)
    plt.yticks(yticks)
    plt.ylim(ylim)
    plt.tight_layout()
    plt.savefig('{}_var_v_phi_errbar.png'.format(protname))
    plt.clf()

    # Var(N) v phi w/o error bars
    plt.plot(dat[:,0]*kt, dat[:,-1], '-o', linewidth=5, markersize=12)
    plt.xlabel(r'$\phi$ (kJ/mol)')
    plt.xticks( np.arange(-2, 14.0, 2.0) )
    plt.xlim(-0.5, 10.5)
    plt.yticks(yticks)
    plt.ylim(ylim)

    plt.tight_layout()
    plt.savefig('{}_var_v_phi.png'.format(protname))
    plt.clf()


