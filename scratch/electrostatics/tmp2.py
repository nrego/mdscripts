import os, glob

fnames_bulk_discharge = sorted(glob.glob('*/bulk_discharge/f_k_all.dat'))
fnames_bound_discharge = sorted(glob.glob('*/bound_discharge/f_k_all.dat'))
fnames_bulk_charge_cav = sorted(glob.glob('*/bulk_charge_cav/f_k_all.dat'))

names = []
bulk_discharge = []
bulk_charge_cav = []
bound_discharge = []

for fname_bulk_discharge, fname_bound_discharge, fname_bulk_charge_cav in zip(fnames_bulk_discharge, fnames_bound_discharge, fnames_bulk_charge_cav):
    
    name = fname_bulk_discharge.split('/')[0]
    assert name == fname_bound_discharge.split('/')[0] == fname_bulk_charge_cav.split('/')[0]
    names.append(name)

    bulk_discharge.append(np.loadtxt(fname_bulk_discharge)[-1])
    bound_discharge.append(np.loadtxt(fname_bound_discharge)[-1])
    bulk_charge_cav.append(np.loadtxt(fname_bulk_charge_cav)[-1])

names = np.array(names)
bulk_discharge = np.array(bulk_discharge)
bound_discharge = np.array(bound_discharge)
bulk_charge_cav = np.array(bulk_charge_cav)