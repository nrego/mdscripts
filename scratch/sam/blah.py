from __future__ import division

import numpy as np
import MDAnalysis


gauss = lambda x, mean, var: (x-mean)**2 / (2*var)
entropy = lambda p1, p2, bc: np.trapz(p1*np.log(p1/p2), bc)

n_v_phi = np.loadtxt('NvPhi.dat')
loghist = np.loadtxt('PvN.dat')

expt_dist = gauss(loghist[:,0], n_v_phi[0,3], n_v_phi[0,4])

#plt.plot(loghist[:,0], loghist[:,1]-loghist[:,1].min())
#plt.plot(loghist[:,0], expt_dist, 'k--')
#plt.show()

act_dist = loghist[:,1] - loghist[:,1].min()
diff = act_dist[0] - expt_dist[0]

np.savetxt('act_diff.dat', np.array([diff]))
'''
bc = loghist[:,0]
hist_expt = np.exp(-expt_dist)
hist_expt /= np.trapz(hist_expt, bc)

hist_act = np.exp(-act_dist)
hist_act /= np.trapz(hist_act, bc)
'''