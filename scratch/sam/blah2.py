# Run after analyze_patches

grid_vals = np.linspace(-1, 1, 5)

perf = np.zeros((grid_vals.size, grid_vals.size, grid_vals.size))
perf[:] = np.inf


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

res = tree_ext.query_ball_tree(tree, r=0.4)
center_mask = np.array([len(arr) for arr in res])
center_mask = center_mask > 0

def gen_w_graph_ext(positions, methyl_mask, wt_mm=1, wt_oo=0, wt_mo=0):
    indices_all = np.arange(36)
    indices_ch3 = indices_all[methyl_mask]
    indices_oh = indices_all[~methyl_mask]

    tree = cKDTree(positions)
    edges = list(tree.query_pairs(r=0.6))

    graph = nx.Graph()
    node_dict = [(idx, dict(phob=methyl_mask[idx])) for idx in indices_all]
    graph.add_nodes_from(node_dict)
    
    for i,j in edges:
        if i in indices_ch3 and j in indices_ch3:
            weight = wt_mm
        elif i in indices_oh and j in indices_oh:
            weight = wt_oo
        else:
            weight = wt_mo

        graph.add_edge(i,j,weight=weight)

    return graph

def gen_keff(wt_mm, wt_oo, wt_mo):
    sum_nodes = np.zeros(methyl_pos.shape[0])
    for idx, methyl_mask in enumerate(methyl_pos):

        ext_methyl_mask = np.zeros(positions_ext.shape[0], dtype=bool)
        ext_methyl_mask[center_mask] = methyl_mask

        w_graph = gen_w_graph(positions, methyl_mask, wt_mm, wt_oo, wt_mo)

        deg = np.array(dict(w_graph.degree(weight='weight')).values())
        sum_nodes[idx] = deg.sum()

    return sum_nodes

for idx_mm, wt_mm in enumerate(grid_vals):
    for idx_oo, wt_oo in enumerate(grid_vals):
        for idx_mo, wt_mo in enumerate(grid_vals):
            print("mm: {} oo: {} mo: {}".format(wt_mm, wt_oo, wt_mo))
            k_eff = gen_keff(wt_mm, wt_oo, wt_mo)

            perf_r2, perf_mse, xvals, pred, reg = fit_general_linear_model(np.array([k_eff, k_vals]).T, energies)

            perf[idx_mm, idx_oo, idx_mo] = perf_mse.mean()


