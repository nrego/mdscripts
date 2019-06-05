import MDAnalysis

univ = MDAnalysis.Universe('top.tpr', 'prot.gro')
targ = univ.select_atoms('segid seg_0_Protein_chain_t and not name H*')
part = univ.select_atoms('segid seg_1_Protein_chain_p and not name H*')

targ.write('targ.gro')
part.write('part.gro')
