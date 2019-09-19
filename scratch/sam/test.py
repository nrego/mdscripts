def idx_from_feat(feat):
    feat = feat.astype(bool)

    return np.arange(36)[feat]


import os, glob

fnames = sorted(glob.glob('*.dat'))

for fname in fnames:
    this_name = fname.split('.')[0]

    feat = idx_from_feat(np.loadtxt(fname))

    newuniv = generate_pattern(univ_oh, univ_ch3, patch_start_idx+feat)

    newuniv.atoms.write("{}.gro".format(this_name))