import matplotlib as mpl

mpl.rcParams.update({'axes.labelsize': 50})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 40})
mpl.rcParams.update({'axes.titlesize': 50})
mpl.rcParams.update({'legend.fontsize':30})

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

ax1.plot(dat[:,0], -dat[:,2], '-or')
ax2.plot(phi_vals, mydist_5, '-ob')

ax1.set_ylabel(r'$-\chi_v$', color='r')
ax1.tick_params('y', colors='r')

ax2.set_ylabel(r'$s$', color='b')
ax2.tick_params('y', colors='b')

ax1.set_xlabel(r'$\phi \; (kJ/mol)$')

plt.show()