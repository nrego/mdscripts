import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

## Generalized bar-chart plotting routines
mpl.rcParams.update({'axes.labelsize': 40})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize': 40})

fig, ax = plt.subplots()
ind = np.arange(28)
width = 1
rects1 = ax.bar(ind[::4], dat[:,0], width=width, edgecolor='k', color='y')
ax.set_ylabel(r'$\Delta \Delta G \; (k_B T)$')
ax.set_xticks(ind[1::4])
labels = ('Ala', 'Val', 'Leu', 'Ile', 'Ser', 'Thr', 'Asn')
ax.set_xticklabels(labels)

rects2 = ax.bar(ind[1::4], dat[:,1], width=width, edgecolor='k', color='g')
rects3 = ax.bar(ind[2::4], dat[:,2], width=width, edgecolor='k', color='b')
ax.legend((rects1[0], rects2[0], rects3[0]), (r'$\Delta \Delta G$', r'$\Delta \Delta G_{\rm{SR}}$', r'$\Delta \Delta G_{\rm{ins}}$'))

