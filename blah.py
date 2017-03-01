import numpy as np

mydat = np.loadtxt('avg_n_n_dep.dat')

from __future__ import division



def get_data(phival, rho_norm):

    
    dirname = 'phi{:0.1f}/'.format(phival)

    n_avg_ndep = np.loadtxt(dirname + 'navg_ndep.dat')
    n_avg = n_avg_ndep[0]
    n_dep = n_avg_ndep[1]

    idx = np.where(mydat[:,0] == phival)
    n_avg_actual = mydat[idx, 1]
    n_dep_actual = mydat[idx, 2]

    #p_vals = np.clip(np.loadtxt(dirname + 'norm_rho_p.dat'), 0.0, 1.0)
    p_vals = np.loadtxt(dirname + 'norm_rho_p.dat')
    

    plevels = np.append(np.linspace(p_vals.min(), 0, 51), np.linspace(0.02,1.0,50), axis=0)
    myres = np.zeros_like(plevels)
    for i, p_thres in enumerate(plevels):
        myres[i] = (p_vals >= p_thres).sum() / p_vals.size

    plt.plot(plevels, myres, '-o', label=r'$\phi={}$'.format(phival))


    return n_avg_actual[0,0], n_dep_actual[0,0], n_avg, n_dep


def get_data_2(phival, rho_norm):

    #dirname = 'phi{:0.1f}/'.format(phival)
    dirname = 'phi{:0.1f}/min_dist2.0/'.format(phival)

    n_avg_ndep = np.loadtxt(dirname + 'navg_ndep.dat')
    n_avg = n_avg_ndep[0]
    n_dep = n_avg_ndep[1]

    idx = np.where(mydat[:,0] == phival)
    n_avg_actual = mydat[idx, 1]
    n_dep_actual = mydat[idx, 2]

    p_vals = np.loadtxt(dirname + 'norm_rho_p.dat')
    plevels = np.append(np.arange(p_vals.min(), 0, 0.02), np.linspace(0.02,1.0,50), axis=0)
    
    myres = np.zeros_like(plevels)

    for i, p_thres in enumerate(plevels):
        myres[i] = (p_vals >= p_thres).sum() / len(p_vals)

    #plt.plot(1-plevels, myres, '-o', label=r'$\phi={}$'.format(phival))
    hist, bb = np.histogram(1-p_vals, bins=100, normed=True)
    plt.plot(bb[:-1]+np.diff(bb)/2.0, hist, label=r'$\phi={}$'.format(phival))

    return n_avg_actual[0,0], n_dep_actual[0,0], n_avg, n_dep
