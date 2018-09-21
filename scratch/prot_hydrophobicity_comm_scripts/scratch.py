import matplotlib as mpl

mpl.rcParams.update({'axes.labelsize': 40})
mpl.rcParams.update({'xtick.labelsize': 80})
mpl.rcParams.update({'ytick.labelsize': 80})
mpl.rcParams.update({'axes.titlesize': 50})
mpl.rcParams.update({'legend.fontsize':30})

dat = np.loadtxt('ntwid_out.dat')

fig, ax = plt.subplots(figsize=(8,8))

ax.plot(dat[:,0], dat[:,1], '-o', linewidth=10, markersize=18)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('Unfavorable Potential')
ax.set_ylabel('Protein Hydration Waters')
fig.tight_layout()
fig.savefig('/Users/nickrego/Desktop/toc.pdf', transparent=True)
plt.show()