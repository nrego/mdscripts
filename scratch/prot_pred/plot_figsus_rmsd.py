from __future__ import division, print_function

import matplotlib as mpl
from matplotlib import rc 
from matplotlib import pyplot as plt
import os, glob
import numpy as np
from constants import k

homedir = os.environ['HOME']

mpl.rcParams.update({'axes.labelsize': 50})
mpl.rcParams.update({'xtick.labelsize': 60})
mpl.rcParams.update({'ytick.labelsize': 60})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':15})

beta = 1 / (300*k)
## Plot rmsd vs phi ##
sys_names = ['2tsc_pred', '1msb_pred', '2mlt', '1ycr_mdm2', '2z59_ubiq']


fig, ax = plt.subplots(figsize=(10,10))

backbone_rmsds = []
heavy_rmsds = []

for sys_name in sys_names:
    dat = np.loadtxt('{}/old_prot_all/bound/rmsd_fit.dat'.format(sys_name))
    
    backbone_rmsds.append(dat[1]/10.0)
    heavy_rmsds.append(dat[2]/10.0) 

indices = np.arange(len(sys_names))

width=0.4
ax.bar(indices-width, backbone_rmsds, width=width, align='edge', label='Backbone')
ax.bar(indices, heavy_rmsds, width, align='edge', label='All Atoms')
ax.set_xticks(indices)
ax.set_xticklabels([])
#fig.legend(loc=1)
#ax.set_xlim(100,110)
#ax.set_ylim(100,110)
#plt.axis('off')

fig.tight_layout()
plt.savefig('{}/Desktop/rmsd.pdf'.format(homedir), transparent=True)

