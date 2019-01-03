from __future__ import division, print_function

from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

def plt_errorbars(bb, vals, errs, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    ax.fill_between(bb, vals-errs, vals+errs, alpha=0.5, facecolor='k', **kwargs)