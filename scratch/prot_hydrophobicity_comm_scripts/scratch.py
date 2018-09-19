dat = np.loadtxt('ntwid_out.dat')

fig, ax = plt.subplots(figsize=(6,5))

ax.plot(dat[:,0], dat[:,1], '-o', linewidth=10, markersize=18)
ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()
fig.savefig('/Users/nickrego/Desktop/toc.pdf', transparent=True)
plt.show()