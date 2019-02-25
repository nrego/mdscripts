# Run after analyze_patches

grid_vals = np.linspace(-1, 1, 5)

perf = np.zeros((grid_vals.size, grid_vals.size, grid_vals.size))
perf[:] = np.inf


def gaus(dist, rcut, A, sigma):

    g = A * np.exp(-dist**2/(2*sigma))
    g[dist>rcut] = 0

    return g

tree = cKDTree(positions)
dists, idxes = tree.query(positions, k=positions.shape[0])

for i, dist in enumerate(dists):
    idx = idxes[i]
    sort_idx = np.argsort(idx)
    dist = dist[sort_idx]
    dist -= 0.5
    dist[i] = np.inf
    dists[i] = dist


def get_gaus(dists, methyl_mask, wt_m, wt_o, wt_mo, sig_m, sig_o, sig_mo, rcutm, rcuto, rcutmo):
    g = np.zeros(methyl_mask.shape[0])

    mm_dists = dists[methyl_mask][:, methyl_mask]
    oo_dists = dists[~methyl_mask][:, ~methyl_mask]
    mo_dists = dists[methyl_mask][:, ~methyl_mask]
    om_dists = dists[~methyl_mask][:, methyl_mask]

    g_meth = gaus(mm_dists, rcutm, wt_m, sig_m).sum(axis=1) + gaus(mo_dists, rcutmo, wt_mo, sig_mo).sum(axis=1)
    g_oh = gaus(oo_dists, rcuto, wt_o, sig_o).sum(axis=1) + gaus(om_dists, rcutmo, wt_mo, sig_mo).sum(axis=1)

    g[methyl_mask] = g_meth
    g[~methyl_mask] = g_oh

    return g


def gen_keff(wt_mm, wt_oo, wt_mo):
    sum_nodes = np.zeros(methyl_pos.shape[0])
    for idx, methyl_mask in enumerate(methyl_pos):

        w_graph = gen_w_graph(positions, methyl_mask, wt_mm, wt_oo, wt_mo)

        deg = np.array(dict(w_graph.degree(weight='weight')).values())
        sum_nodes[idx] = deg.sum()

    return sum_nodes

params = []
all_perf = []
for idx_mm, wt_mm in enumerate(grid_vals):
    for idx_oo, wt_oo in enumerate(grid_vals):
        for idx_mo, wt_mo in enumerate(grid_vals):
            print("mm: {} oo: {} mo: {}".format(wt_mm, wt_oo, wt_mo))
            k_eff = gen_keff(wt_mm, wt_oo, wt_mo)

            perf_r2, perf_mse, xvals, pred, reg = fit_general_linear_model(np.array([k_eff, k_vals]).T, energies)

            perf[idx_mm, idx_oo, idx_mo] = perf_mse.mean()
            all_perf.append(perf_mse.mean())
            params.append((wt_mm, wt_oo, wt_mo))


width_vals = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5])
rcut_vals = np.array([0.2, 3.6])
this_param = []
this_perf = []
for idx_meth_wt, meth_wt in enumerate([1]):
    print("meth wt: {}".format(meth_wt))
    for idx_oh_wt, oh_wt in enumerate(grid_vals):
        print('oh wt: {}'.format(oh_wt))
        for idx_mo_wt, mo_wt in enumerate(grid_vals):
            for idx_meth_width, meth_width in enumerate(width_vals):
                for idx_mo_width, mo_width in enumerate(width_vals):
                    for idx_oh_width, oh_width in enumerate(width_vals):
                        for idx_rcut, rcutm in enumerate(rcut_vals):
                            for idx2, rcuto in enumerate(rcut_vals):
                                for idx3, rcutmo in enumerate(rcut_vals):
                                    sum_nodes = np.zeros(methyl_pos.shape[0])
                                    for idx, methyl_mask in enumerate(methyl_pos):
                                        sum_nodes[idx] = get_gaus(dists, methyl_mask, meth_wt, oh_wt, mo_wt, meth_width, oh_width, mo_width, rcutm, rcuto, rcutmo).sum()
                                    perf_r2, perf_mse, xvals, pred, reg = fit_general_linear_model(np.array([sum_nodes, k_vals]).T, energies)
                                    this_param.append((meth_wt, oh_wt, mo_wt, meth_width, oh_width, mo_width, rcutm, rcuto, rcutmo))
                                    this_perf.append(perf_mse.mean())

this_perf = np.array(this_perf)

''' 
## Generate grid of center points
z_space = 0.5 # 0.5 nm spacing
y_space = np.sqrt(3)/2.0 * z_space

pos_row = np.array([-0.5, 0. , 0.5, 1. , 1.5, 2. , 2.5, 3., 3.5, 4.0])
#y_pos = 0
y_pos = -y_space
positions_ext = []
for i in range(8):
    if i % 2 != 0:
        this_pos_row = pos_row
    else:
        this_pos_row = pos_row + z_space/2.0

    for j in range(8):
        z_pos = this_pos_row[j]
        positions_ext.append(np.array([y_pos, z_pos]))

    y_pos += y_space

positions_ext = np.array(positions_ext)
N = positions_ext.shape[0]

tree_ext = cKDTree(positions_ext)
tree = cKDTree(positions)
'''