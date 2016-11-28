from __future__ import division, print_function
import numpy as np
from westext.stringmethod import DefaultStringMethod
import scipy


def dist(pt1, pt2):
    return np.sum((pt1-pt2)**2)

def mueller(x, y):
    aa = [-1, -1, -6.5, 0.7]
    bb = [0, 0, 11, 0.6]
    cc = [-10, -10, -6.5, 0.7]
    AA = [-200, -100, -170, 15]

    XX = [1, 0, -0.5, -1]
    YY = [0, 0.5, 1.5, 1]
    V1 = 0
    for j in range(4):
        V1 += AA[j] * np.exp(aa[j] * (x - XX[j])**2 + \
              bb[j] * (x - XX[j]) * (y - YY[j]) + cc[j] * (y - YY[j])**2)

    return V1

def calculate_length(x):
    dd = x - np.roll(x, 1, axis=0)
    dd[0,:] = 0.0
    return np.cumsum(np.sqrt((dd*dd).sum(axis=1)))

##Params
beta = 1 
dtau = 0.1
kappa = 0.1
n_centers = 20

kappa_n = kappa*n_centers*dtau

# A is banded matrix on r.h.s of eqn: A i_star[k] = b[k]
#   for applying smoothing term r_star
#   b[k] is calculated difference vector, from b[k] = centers[k] - avg_pos[k],
#   k elem (0,1)
row1 = np.zeros(n_centers)
row1[2:] = - kappa_n
row3 = np.zeros(n_centers)
row3[:-2] = - kappa_n
row2 = np.ones(n_centers)
row2[1:-1] = 1 + 2*kappa_n
A = np.matrix((row1, row2, row3))

centers = np.zeros((n_centers, 2))
centers[:,0] = np.linspace(-1.5,1.2,20)
centers[:,1] = 1.0

string = DefaultStringMethod(centers, [n_centers])

xx, yy = np.mgrid[-1.5:1.2:0.01, -0.2:2.0:0.01]

assert xx.shape == yy.shape
nx = xx.shape[0]
ny = xx.shape[1]

energy = mueller(xx, yy)


n_iter = 500
centers_record = np.empty((n_iter+1, n_centers, 2))
centers_record[0,...] = centers.copy()
rmsd = np.zeros(n_iter)

for i_iter in range(n_iter):
    print("Running Iter {} of {}".format(i_iter+1, n_iter))
    avg_pos = np.zeros_like(centers)
    x_indices = [ [] for i in range(n_centers) ]
    y_indices = [ [] for i in range(n_centers) ]

    for xidx in range(nx):
        for yidx in range(ny):

            pt = np.array([xx[xidx,yidx],yy[xidx,yidx]])

            mindist = float('inf')
            center_assign = -1

            for center_idx, centerpt in enumerate(centers):
                d = dist(pt, centerpt)
                if d < mindist:
                    mindist = d
                    center_assign = center_idx

            x_indices[center_assign].append(xidx)
            y_indices[center_assign].append(yidx)

    for i in range(n_centers):
        # array of energies for all gridpoints in voronoi cell i
        cell_energy = energy[x_indices[i], y_indices[i]]
        #cell_energy -= cell_energy.max()

        probs = np.exp(-beta*cell_energy)
        probs /= probs.sum()

        cell_pts = np.array([xx[x_indices[i], y_indices[i]], yy[x_indices[i], y_indices[i]]]).T

        avg_pos[i] = probs.dot(cell_pts)

    centers_new = np.zeros_like(centers)
    b = centers - dtau*(centers-avg_pos)

    # Smooth string
    for k in range(2):
        centers_new[:,k] = scipy.linalg.solve_banded((1,1), A, b[:,k])

    # Enforce equal spacing between new centers
    L = calculate_length(centers_new)
    L /= L[-1]
    g2 = np.linspace(0,1,n_centers)

    for k in range(2):
        f = scipy.interpolate.interp1d(L,centers_new[:,k],kind='linear')
        centers_new[:,k] = f(g2)

    rmsd[i_iter] = np.sum((centers - centers_new)**2) / n_centers
    centers = centers_new
    centers_record[i_iter+1, ...] = centers.copy()

