import MDAnalysis
import numpy as np

import os, glob

from constants import k

beta = 1/(k*300)

water_data_0 = np.load('phi_000/rho_data_dump_rad_6.0.dat.npz')['rho_water'].T
water_ref_0 = water_data_0.mean(axis=1)
fnames = sorted(glob.glob('phi_*/rho_data_dump_rad_6.0.dat.npz'))

buried_mask = np.loadtxt('../buried_mask.dat').astype(bool)

univ = MDAnalysis.Universe('phi_000/dynamic_volume_water_avg.pdb')

leu07 = univ.select_atoms('resid 7 and not name H*')[2:6]
leu19 = univ.select_atoms('resid 19 and not name H*')[2:6]
leu21 = univ.select_atoms('resid 21 and not name H*')[2:6]
leu63 = univ.select_atoms('resid 63 and not name H*')[2:6]

bb = np.linspace(0,2,25)
avg_rho07 = []
avg_rho19 = []
avg_rho21 = []
avg_rho63 = []

var_rho07 = []
var_rho19 = []
var_rho21 = []
var_rho63 = []

phi_vals = []
for fname in fnames[:-4]:
    dirname = os.path.dirname(fname)
    phi_val = float(dirname.split('_')[-1]) / 10.
    phi_vals.append(phi_val)

    water_data = np.load(fname)['rho_water'].T
    water_norm = water_data / water_ref_0[:,None]

    norm07 = water_norm[leu07.indices, :]
    norm19 = water_norm[leu19.indices, :]
    norm21 = water_norm[leu21.indices, :]
    norm63 = water_norm[leu63.indices, :]

    #hist07, bb = np.histogram(norm07.flatten(), bins=bb, normed=True) 
    #hist19, bb = np.histogram(norm19.flatten(), bins=bb, normed=True) 
    #hist21, bb = np.histogram(norm21.flatten(), bins=bb, normed=True) 
    #hist63, bb = np.histogram(norm63.flatten(), bins=bb, normed=True) 

    #plt.plot(bb[:-1], hist07, label='L07')
    #plt.plot(bb[:-1], hist19, label='L19')
    #plt.plot(bb[:-1], hist21, label='L21')
    #plt.plot(bb[:-1], hist63, label='L63')

    print("phi: {}".format(dirname))
    print("  leu07: {}".format(norm07.mean()))
    print("  leu19: {}".format(norm19.mean()))
    print("  leu21: {}".format(norm21.mean()))
    print("  leu63: {}".format(norm63.mean()))

    avg_rho07.append(norm07.mean())
    avg_rho19.append(norm19.mean())
    avg_rho21.append(norm21.mean())
    avg_rho63.append(norm63.mean())

    var_rho07.append(norm07.var())
    var_rho19.append(norm19.var())
    var_rho21.append(norm21.var())
    var_rho63.append(norm63.var())

    #plt.legend()
    #plt.show()

phi_vals = np.array(phi_vals)
plt.plot(beta*phi_vals, avg_rho07, '-o', label='L07', linewidth=8, markersize=18)
#plt.plot(beta*phi_vals, avg_rho19, '-o', label='L19')
plt.plot(beta*phi_vals, avg_rho21, '-o', label='L21', linewidth=8, markersize=18)
plt.plot(beta*phi_vals, avg_rho63, '-o', label='L63', linewidth=8, markersize=18)

plt.legend()
plt.show()



plt.plot(beta*phi_vals, var_rho07, '-o', label='L07', linewidth=8, markersize=18)
#plt.plot(beta*phi_vals, var_rho19, '-o', label='L19')
plt.plot(beta*phi_vals, var_rho21, '-o', label='L21', linewidth=8, markersize=18)
plt.plot(beta*phi_vals, var_rho63, '-o', label='L63', linewidth=8, markersize=18)

plt.legend()
plt.show()


fig, ax = plt.subplots()
ddgs = np.array([ 3.753219,  5.449717,  7.631214])

indices = np.arange(3)
width = 1
ax.bar(indices, ddgs, width=width, edgecolor='k')
ax.set_xticks(indices)
ax.set_xticklabels(['L07S', 'L21S', 'L63S'])
ax.set_ylabel(r'$\beta \phi$')



