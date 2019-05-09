from __future__ import division, print_function

import os, glob
from scipy import special
from util import *

import matplotlib as mpl

fnames = glob.glob('pattern_sample/*/d_*/trial_0/PvN.dat')
fnames = np.append(fnames, glob.glob('inv_pattern_sample/*/d_*/trial_0/PvN.dat'))

nvphis = []
pvns = []
phistars = []
fes = []
ndats = []
kvals = []
peak_vars = []
for fname in fnames:

    is_k = fname.split('/')[1].split('_')[0] == 'k'
    number = int(fname.split('/')[1].split('_')[1])
    if is_k:
        kval = number
    else:
        kval = 36 - number

    kvals.append(kval)

    dirname = os.path.dirname(fname)
    nvphi = np.loadtxt('{}/NvPhi.dat'.format(dirname))
    #ndat = np.loadtxt('{}/n_out.dat'.format(dirname))
    pvn = np.loadtxt(fname)

    nvphis.append(nvphi)
    pvns.append(pvn)
    fe = pvn[0,1]
    fes.append(fe)
    phistar_idx = np.argmax(nvphi[:,4])
    var_0 = nvphi[0, 4]
    peak_var = nvphi[phistar_idx, 4]
    peak_vars.append(peak_var/var_0)
    phistar = nvphi[phistar_idx, 0]

    phistars.append(phistar)

    #ndats.append(ndat)

phistars = np.array(phistars)
fes = np.array(fes)
nvphis = np.array(nvphis)
ndats = np.array(ndats)

peak_vars = np.array(peak_vars)
sort_idx = np.argsort(phistars)
plt.close('all')

norm = plt.Normalize(1, np.ceil(peak_vars.max()))
cmap = mpl.cm.jet
sc = plt.scatter(phistars, fes, c=peak_vars, cmap=cmap, norm=norm)
plt.colorbar(sc)



plt.close('all')
for idx in [0,5,24]:
    
    this_phistar = phistars[sort_idx][idx]
    this_fe = fes[sort_idx][idx]
    this_nvphi = nvphis[sort_idx][idx]
    this_fname = fnames[sort_idx][idx]
    this_ndat = ndats[sort_idx][idx]

    print("fname: {}  phistar: {:0.2f}  fe: {:0.2f}".format(this_fname, this_phistar, this_fe))

    plt.plot(this_nvphi[:,0], this_nvphi[:,4], label=r'$\phi^*={:1.2f}$'.format(this_phistar))
    #plt.plot(this_ndat[:,0], this_ndat[:,2], '-o', label=r'$\phi^*={:1.2f}$'.format(this_phistar))


