import numpy as np

from __future__ import division, print_function
import os, glob


dirnames = ['phobic', 'polar_ring', 'charged']

phobic_dg_pw = 0.0
phobic_dg_pp = 0.0
phobic_bulk_cav = np.loadtxt('phobic/bulk_cav/PvN.dat')[0,1]
phobic_n_v_phi = np.loadtxt('phobic/bulk_cav/NvPhi.dat')


polar_dg_pw = np.loadtxt('polar_ring/bound_discharge/f_k_all.dat')[-1]
polar_dg_pp = np.loadtxt('polar_ring/bulk_discharge/f_k_all.dat')[-1]
polar_bulk_cav = np.loadtxt('polar_ring/bulk_cav/PvN.dat')[0,1]
polar_n_v_phi = np.loadtxt('polar_ring/bulk_cav/NvPhi.dat')

charge_dg_pw = np.loadtxt('charged/bound_discharge/f_k_all.dat')[-1]
charge_dg_pp = np.loadtxt('charged/bulk_discharge/f_k_all.dat')[-1]
charge_bulk_cav = np.loadtxt('charged/bulk_cav/PvN.dat')[0,1]
charge_n_v_phi = np.loadtxt('charged/bulk_cav/NvPhi.dat')


plt.plot(phobic_n_v_phi[:,0], phobic_n_v_phi[:,1], label='hydrophobic')
plt.plot(polar_n_v_phi[:,0], polar_n_v_phi[:,1], label='polar ring')
plt.plot(charge_n_v_phi[:,0], charge_n_v_phi[:,1], label='charged center')
plt.legend()

plt.plot(phobic_n_v_phi[:,0], phobic_n_v_phi[:,2], label='hydrophobic')
plt.plot(polar_n_v_phi[:,0], polar_n_v_phi[:,2], label='polar ring')
plt.plot(charge_n_v_phi[:,0], charge_n_v_phi[:,2], label='charged center')
plt.legend()