import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

## Generalized bar-chart plotting routines
mpl.rcParams.update({'axes.labelsize': 40})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize': 40})

fig, ax = plt.subplots()



names = ('Gly', 'Ala', 'Val', 'Leu', 'Ile', 'Ser', 'Thr', 'Asn')
n_bars = len(names)
n_comp = 3
n_comp += 1 
ind = np.arange(n_bars*n_comp)
width = 1
rects1 = ax.bar(ind[::n_comp], dat_alc, width=width, edgecolor='k', color='b', label='Alchemical', yerr=err_alc)
rects2 = ax.bar(ind[1::n_comp], dat_sh_4.sum(axis=1), width=width, edgecolor='k', color='r', label='QCT, $r=4.0 \AA$', yerr=np.sqrt(np.sum(err_sh_4**2, axis=1)))
rects3 = ax.bar(ind[2::n_comp], dat_sh_4.sum(axis=1), width=width, edgecolor='k', color='g', label='QCT, $r=6.0 \AA$', yerr=np.sqrt(np.sum(err_sh_6**2, axis=1)))
ax.set_ylabel(r'$ \Delta G\; (k_B T)$')
ax.set_xticks(ind[1::n_comp])
#names = ('gly', 'ala', 'val', 'leu', 'ile', 'ser', 'thr', 'asn')

ax.set_xticklabels(names)
ax.legend()

plt.show()

fig, ax = plt.subplots()

rects1 = ax.bar(ind[::n_comp], dat_sh_4[:,0], width=width, edgecolor='k', color='r', label=r'fill', yerr=err_sh_4[:,0])
rects2 = ax.bar(ind[1::n_comp], dat_sh_4[:,1], width=width, edgecolor='k', color='b', label=r'cav', yerr=err_sh_4[:,1])
rects3 = ax.bar(ind[2::n_comp], dat_sh_4[:,2], width=width, edgecolor='k', color='g', label=r'ins', yerr=err_sh_4[:,2])
ax.set_ylabel(r'$ \Delta G\; (k_B T)$')
ax.set_xticks(ind[1::n_comp])

ax.set_xticklabels(names)
ax.legend()

plt.show()

fig, ax = plt.subplots()

rects1 = ax.bar(ind[::n_comp], dat_sh_6[:,0], width=width, edgecolor='k', color='r', label=r'fill', yerr=err_sh_6[:,0])
rects2 = ax.bar(ind[1::n_comp], dat_sh_6[:,1], width=width, edgecolor='k', color='b', label=r'cav', yerr=err_sh_6[:,1])
rects3 = ax.bar(ind[2::n_comp], dat_sh_6[:,2], width=width, edgecolor='k', color='g', label=r'ins', yerr=err_sh_6[:,2])
ax.set_ylabel(r'$ \Delta G\; (k_B T)$')
ax.set_xticks(ind[1::n_comp])

ax.set_xticklabels(names)
ax.legend()

plt.show()

## Plot dg_sr, dg_ins for r=4 A
fig, ax = plt.subplots()

n_comp = 2
n_comp += 1 
ind = np.arange(n_bars*n_comp)

rects1 = ax.bar(ind[::n_comp], dat_sh_4[:,0]+dat_sh_4[:,1], width=width, edgecolor='k', color='m', label=r'cav+fill', yerr=np.sqrt(np.sum(err_sh_4[:,:2]**2, axis=1)))
rects2 = ax.bar(ind[1::n_comp], dat_sh_4[:,2], width=width, edgecolor='k', color='g', label=r'ins', yerr=err_sh_4[:,2])
ax.set_ylabel(r'$ \Delta G\; (k_B T)$')
ax.set_xticks(ind[1::n_comp]-0.5)

ax.set_xticklabels(names)
ax.legend()

plt.show()


## Plot dg_sr, dg_ins for r=6 A
fig, ax = plt.subplots()

n_comp = 2
n_comp += 1 
ind = np.arange(n_bars*n_comp)

rects1 = ax.bar(ind[::n_comp], dat_sh_6[:,0]+dat_sh_6[:,1], width=width, edgecolor='k', color='m', label=r'cav+fill', yerr=np.sqrt(np.sum(err_sh_6[:,:2]**2, axis=1)))
rects2 = ax.bar(ind[1::n_comp], dat_sh_6[:,2], width=width, edgecolor='k', color='g', label=r'ins', yerr=err_sh_6[:,2])
ax.set_ylabel(r'$ \Delta G\; (k_B T)$')
ax.set_xticks(ind[1::n_comp]-0.5)

ax.set_xticklabels(names)
ax.legend()

plt.show()


## DDG w.r.t glycine, etc -> explore additivity/predictions based on delta G for each group
# ddg w.r.t glycine -> additivity?
ddg_alc = (dat_alc - dat_alc[0])[1:]

ddg_sh_4 = (dat_sh_4 - dat_sh_4[0])[1:]
ddg_sh_6 = (dat_sh_6 - dat_sh_6[0])[1:]

dg_meth = ddg_alc[0]
dg_hydr = ddg_alc[4] - dg_meth

dg_meth_sh_4 = ddg_sh_4[0]
dg_hydr_sh_4 = ddg_sh_4[4] - dg_meth_sh_4

dg_meth_sh_6 = ddg_sh_6[0]
dg_hydr_sh_6 = ddg_sh_6[4] - dg_meth_sh_6

# Alanine and Serine will be exact, by definition
pred = np.array([dg_meth, 3*dg_meth, 4*dg_meth, 4*dg_meth, dg_meth+dg_hydr, 2*dg_meth+dg_hydr])

## Plot DDG's w.r.t glycine
fig, ax = plt.subplots()

names = ('ala', 'val', 'leu', 'ile', 'ser', 'thr', 'asn')
n_bars = len(names)

n_comp = 1
#n_comp += 1 
ind = np.arange(n_bars*n_comp)

rects1 = ax.bar(ind[::n_comp], ddg_alc, width=width, edgecolor='k', color='c', yerr=np.sqrt(err_alc**2 + err_alc[0]**2)[1:])
ax.set_ylabel(r'$ \Delta G\; (k_B T)$')
ax.set_xticks(ind)

ax.set_xticklabels(names)
ax.legend()

plt.show()



