import numpy as np
import MDAnalysis

import argparse
from IPython import embed
import os, glob

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np

from scratch.sam.util import *

import itertools

# For SAM schematic pattern plotting
figsize = (10,10)

mpl.rcParams.update({'axes.labelsize': 45})
mpl.rcParams.update({'xtick.labelsize': 50})
mpl.rcParams.update({'ytick.labelsize': 50})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':34})

homedir = os.environ['HOME']

########################
# Plot pattern with edges types indicated
########################

plt.close('all')

new_state = State(np.delete(state.pt_idx, [6,7,12,15,20]))
#new_state.plot(mask=[0,63])
new_state.plot()

symbols = np.array(['' for i in range(state.N_tot)], dtype=object)
symbols[state.nodes_buried] = 'ko'
#symbols[state.nodes_peripheral] = 'yP'
symbols[state.nodes_peripheral] = 'ko'

styles = np.array(['' for i in range(state.n_edges)], dtype=object)
styles[[31,51,52,54]] = ':'
styles[33] = '-.'
styles[32] = '--'
styles[[63,64]] = '-'

widths = np.array([0 for i in range(state.n_edges)])
widths[[31,32,33,51,52,54,63,64]] = 3

state.plot_edges(symbols=symbols, line_widths=widths, line_styles=styles)

plt.savefig("{}/Desktop/fig_nodetype".format(homedir), transparent=True)


## Do the legend ##
plt.close('all')
plt.plot([0,0], [0,0], 'ko', markersize=20, label='buried')
plt.plot([0,0], [0,0], 'yP', markersize=20, label='peripheral')
plt.plot([0,0], [0,0], 'rX', markersize=20, label='external')

plt.xlim(100,200)
plt.legend(loc='center')

plt.axis('off')
plt.savefig("{}/Desktop/leg".format(homedir), transparent=True)

## Do the legend ##
plt.close('all')
plt.plot([0,0], [0,0], 'k:', linewidth=3, label='    edge')
plt.plot([0,0], [0,0], 'k-.', linewidth=3, label='    edge')
plt.plot([0,0], [0,0], 'k--', linewidth=3, label='    edge')
plt.plot([0,0], [0,0], 'k-', linewidth=3, label='    edge')

plt.xlim(100,200)
plt.legend(loc='center')

plt.axis('off')
plt.savefig("{}/Desktop/leg_edge".format(homedir), transparent=True)


