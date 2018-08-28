import os, glob

from constants import k
import matplotlib as mpl

mpl.rcParams.update({'axes.labelsize': 20})
mpl.rcParams.update({'xtick.labelsize': 20})
mpl.rcParams.update({'ytick.labelsize': 20})
mpl.rcParams.update({'axes.titlesize': 25})
mpl.rcParams.update({'legend.fontsize':18})

kt = 300*k
beta = 1 / kt



def window_smoothing(arr, windowsize=10):
    ret_arr = np.zeros_like(arr)
    ret_arr[:] = float('nan')

    for i in range(ret_arr.size):
        ret_arr[i] = arr[i-windowsize:i+windowsize].mean()

    return ret_arr


fnames = sorted( glob.glob('sh_*/neglogpdist.dat') )
phi_vals = beta*np.arange(0, 10, 0.1)

for f in reversed(fnames):
    dirname = os.path.dirname(f)

    #rv = int(dirname.split('_')[-1]) / 10
    #title_header = '/Users/nickrego/Desktop/ubiq_rv_{:02d}'.format(rv)
    #title_header = '/Users/nickrego/Desktop/bulk_rv_10'
    title_header = '/Users/nickrego/Desktop/{}'.format(dirname)
    rv=0
    dat = np.loadtxt(f)

    n_vals = dat[:,0]
    neglogpdist = dat[:,1]

    window = 5
    smooth = window_smoothing(neglogpdist, windowsize=window)

    fin_diff = np.diff(smooth)
    
    plt.plot(n_vals, neglogpdist)
    plt.xlabel(r'$N$')
    plt.ylabel(r'$\beta F_V(N)$')
    plt.tight_layout()
    plt.savefig('{}_F.pdf'.format(title_header))
    plt.show()

    plt.plot(n_vals[1:], fin_diff)
    plt.xlabel(r'$N$')
    plt.ylabel(r'$\frac{\partial \beta F_V(N)}{\partial N}$')
    ymin, ymax = plt.ylim()
    plt.ylim(ymin, 0)
    plt.tight_layout()
    plt.savefig('{}_Fprime.pdf'.format(title_header))
    plt.show()

    curv = np.diff(window_smoothing(fin_diff, 5))
    plt.plot(n_vals[2:], curv)
    plt.xlabel(r'$N$')
    plt.ylabel(r'$\frac{\partial^2 \beta F_V(N)}{\partial N^2}$')
    plt.tight_layout()
    plt.savefig('{}_Fcurv.pdf'.format(title_header))
    plt.show()

    avg_ns = []
    var_ns = []
    for phi in phi_vals:
        bias = n_vals*phi + neglogpdist
        bias -= bias.min()
        pdist = np.exp(-bias)
        pdist /= pdist.sum()

        avg_n = np.trapz(pdist*n_vals, n_vals)
        avg_nsq = np.trapz(pdist*n_vals**2, n_vals)
        var_n = avg_nsq - avg_n**2

        avg_ns.append(avg_n)
        var_ns.append(var_n)

    avg_ns = np.array(avg_ns)
    var_ns = np.array(var_ns)


    max_idx = np.argmax(var_ns)

    phi_star = phi_vals[max_idx]

    plt.plot(phi_vals[:max_idx+20], avg_ns[:max_idx+20])
    plt.xlabel(r'$\beta \phi$')
    plt.ylabel(r'$\langle N_V \rangle_\phi$')
    plt.tight_layout()
    plt.savefig('{}_n_v_phi.pdf'.format(title_header))
    plt.show()

    plt.plot(phi_vals[:max_idx+20], var_ns[:max_idx+20])
    plt.xlabel(r'$\beta \phi$')
    plt.ylabel(r'$\langle \delta N^2_V \rangle_\phi$')
    plt.tight_layout()
    plt.savefig('{}_sus_v_phi.pdf'.format(title_header))
    plt.show()

    bias = n_vals*phi_star + neglogpdist
    bias -= bias.min()

    plt.plot(n_vals, neglogpdist, label=r'$\beta \phi=0$')
    plt.plot(n_vals, bias, label=r'$\beta \phi^*={:0.2f}$'.format(phi_star))
    plt.legend()
    plt.ylim(0,12)
    plt.xlabel(r'$N$')
    plt.ylabel(r'$\beta F^\phi(N)$')
    plt.tight_layout()
    plt.savefig('{}_Fphistar.pdf'.format(title_header))
    plt.show()

#plt.legend()
#plt.show()