from matplotlib import rc
import os, glob

rc('text', usetex=False)

fnames = sorted(glob.glob('run_*/dhdl.xvg'))

for fname in fnames:
    dat = np.loadtxt(fname, comments=['@','#'])
    name = os.path.dirname(fname)
    plt.plot(dat[:,1], '-o', label=name)

plt.legend()
plt.show()

