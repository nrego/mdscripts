from scipy.spatial import cKDTree
import numpy as np
import MDAnalysis

univ = MDAnalysis.Universe('../bulk_discharge/contacts.pdb')
contact_mask = univ.atoms.tempfactors == 1

univ = MDAnalysis.Universe('phi_200/confout.gro')
univ.add_TopologyAttr('tempfactors')
prot = univ.select_atoms('protein')
prot[contact_mask].tempfactors = 1
contact = prot[contact_mask].select_atoms('not name H*')
ow = univ.select_atoms('name OW')

tree_contact = cKDTree(contact.positions)
tree_ow = cKDTree(ow.positions)

res = tree_contact.query_ball_tree(tree_ow, r=6.0)
close_ow_ids = np.unique( np.concatenate(res) ).astype(int)
sel_str = ''.join([' {}'.format(idx) for idx in ow[close_ow_ids].ids])

close_ow = univ.select_atoms('bynum {}'.format(sel_str))
close_water_ids = np.ravel([[idx, idx+1, idx+2] for idx in close_ow.ids])

sel_str = ''.join([' {}'.format(idx) for idx in close_water_ids])
close_waters = univ.select_atoms('bynum {}'.format(sel_str))
close_waters.write('close_waters.pdb')

empty = univ.select_atoms('not (bynum {})'.format(sel_str))
empty.write('empty.pdb')
empty.write('empty.gro')