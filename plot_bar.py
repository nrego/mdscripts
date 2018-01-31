import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

## Generalized bar-chart plotting routines
mpl.rcParams.update({'axes.labelsize': 40})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize': 40})

fig, ax = plt.subplots()
ind = np.arange(4)
width = 1
rects1 = ax.bar(ind, dat, width=width, edgecolor='k', color='b')
ax.set_ylabel(r'$ \Delta G_{\rm{sr}}\; (k_B T)$')
ax.set_xticks(ind)
#names = ('gly', 'ala', 'val', 'leu', 'ile', 'ser', 'thr', 'asn')
names = ('Wt', 'Cent', 'Ring', 'Edge')
ax.set_xticklabels(names)
#ax.legend((rects1[0], rects2[0]), ('bind', 'cav'))

rects2 = ax.bar(ind[1::4], dat[:,1], width=width, edgecolor='k', color='g')
rects3 = ax.bar(ind[2::4], dat[:,2], width=width, edgecolor='k', color='b')
ax.legend((rects1[0], rects2[0], rects3[0]), (r'$\Delta \Delta G$', r'$\Delta \Delta G_{\rm{SR}}$', r'$\Delta \Delta G_{\rm{ins}}$'))

