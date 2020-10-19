
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


state = State(np.delete(np.arange(36), [6,7,12,15,20]))



##########################################
# Plot pattern with edges types indicated
##########################################

plt.close('all')

state.plot()
symbols = np.array(['ko' for i in range(state.N_tot)], dtype=object)

styles = np.array(['' for i in range(state.n_edges)], dtype=object)
styles[state.edges_int_indices] = '-'
styles[state.edges_ext_indices] = ':'
widths = np.array([3 for i in range(state.n_edges)])

annotation_color = np.array(['darkorange' for i in range(state.n_edges)])
annotation_color[state.edges_ext_indices] = 'fuchsia'

state.plot_edges(symbols=symbols, line_widths=widths, line_styles=styles, do_annotate=True, annotation_color=annotation_color)

##################################################
# Plot pattern with selected edges types indicated
##################################################

plt.close('all')

annotation_size = np.array(['x-large' for i in range(state.N_tot)], dtype=object)
annotation_size[[6,12,13,18]] = 'xx-large'
annotation_color = ['y' for i in range(state.N_tot)]
state.plot(do_annotate=True, annotation_size=annotation_size, annotation_color=annotation_color)

symbols = np.array(['ko' for i in range(state.N_tot)], dtype=object)


styles = np.array(['' for i in range(state.n_edges)], dtype=object)
styles[[51,52,54]] = ':'
styles[[33,53,55]] = '-'

widths = np.array([0 for i in range(state.n_edges)])
widths[styles!=''] = 3

state.plot_edges(symbols=symbols, line_widths=widths, line_styles=styles)

plt.savefig("{}/Desktop/fig_nodetype".format(homedir), transparent=True)


## Do the legend ##
plt.close('all')
plt.plot([0,0], [0,0], 'k-', linewidth=3, label='internal pair')
plt.plot([0,0], [0,0], 'k:', linewidth=3, label='external pair')


plt.xlim(100,200)
plt.legend(loc='center')

plt.axis('off')
plt.savefig("{}/Desktop/leg_edge".format(homedir), transparent=True)


