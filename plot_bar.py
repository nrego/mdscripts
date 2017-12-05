import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

## Generalized bar-chart plotting routines
mpl.rcParams.update({'axes.labelsize': 40})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize': 40})



fig, ax = plt.subplots()
ind = np.arange(3)
width = 1
rects1 = ax.bar(ind, ddg_cav, width=width, edgecolor='k', color='b')
ax.set_ylabel(r'$ \Delta G_{\rm{cav}} \; (k_B T)$')
ax.set_xticks(ind)
ax.set_xticklabels(('L07S', 'L19S', 'L21S'))
#ax.legend((rects1[0], rects2[0]), ('bind', 'cav'))

#rects2 = ax.bar(ind+width, cav_unbound[1:] - cav_unbound[0], width=width, edgecolor='k', color='r')

