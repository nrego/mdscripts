from __future__ import division, print_function; __metaclass__ = type
import numpy as np

import MDAnalysis
from IPython import embed


def fib(N, N_tot):
    assert N <= N_tot

    if N == (N_tot+1)//2:
        return 0

    else:
        return (N-1)+fib(N-1, N_tot)


def gen_plate_position(n_mid=3, d=3):

    if n_mid % 2 == 0:
        raise ValueError("Only odd numbered lattices currently supported")
    n_edge = fib(n_mid, n_mid)
    n_atoms = n_mid + 2*n_edge

    rad = (n_mid - 1)*d/2.0
    height = (rad/2.0)*np.sqrt(3)

    y_offset = np.sqrt(3)*0.5*d
    all_indices = np.arange(n_atoms)
    positions = np.zeros((n_atoms, 3))

    pos_idx = 0

    center_pt_idx = n_mid // 2 
    # slices bounded by two edge verts and center point
    # order is always upper left, lower left, upper right, lower right, upper center, lower center
    vert_slice_dict = []
    # indices in each "slice"
    slices = [[], [], [], [], [], []]

    # indices of center 
    center_indices = []

    # Use these lists as stacks to keep track of vertex pairs
    left_verts_top = list()
    left_verts_bot = list()
    right_verts_top = list()
    right_verts_bot = list()
    for i, n_row in enumerate(range(n_mid,(n_mid-1)//2,-1)):
        y_pos = i*y_offset

        # this row has verts if it's the first or last row
        vert_row = (i == 0) or (i == (n_mid - 1)//2)

        for j in range(n_row):
            if j > 0 and j < n_row-1 and n_row-2 > 2:
                center_indices.append(pos_idx)
                if i > 0:
                    center_indices.append(pos_idx+n_edge)
            x_offset = i*0.5*d
            x_pos = x_offset + j*d

            positions[pos_idx] = x_pos, y_pos, 0.0
            if i > 0:
                positions[pos_idx+n_edge] = x_pos, -y_pos, 0.0

            ## account for verts
            if vert_row:
                # left vert
                if j == 0:
                    # first row - just append
                    if i == 0:
                        left_verts_top.append(pos_idx)
                        left_verts_bot.append(pos_idx)
                    # last row
                    else:
                        prev_vert_top = left_verts_top.pop()
                        prev_vert_bot = left_verts_bot.pop()
                        vert_slice_dict.append((pos_idx, prev_vert_top))
                        vert_slice_dict.append((pos_idx+n_edge, prev_vert_bot))
                        left_verts_top.append(pos_idx)
                        left_verts_bot.append(pos_idx+n_edge)
                # right vert
                elif j == n_row-1:
                    # first row - just append
                    if i == 0:
                        right_verts_top.append(pos_idx)
                        right_verts_bot.append(pos_idx)
                    # last row
                    else:
                        prev_vert_top = right_verts_top.pop()
                        prev_vert_bot = right_verts_bot.pop()
                        vert_slice_dict.append((pos_idx, prev_vert_top))
                        vert_slice_dict.append((pos_idx+n_edge, prev_vert_bot))

                        prev_vert_top = left_verts_top.pop()
                        prev_vert_bot = left_verts_bot.pop()
                        vert_slice_dict.append((prev_vert_top, pos_idx))
                        vert_slice_dict.append((prev_vert_bot, pos_idx+n_edge))

            # Add this position to appropriate slice
            if i != 0:
                if x_pos <= rad:
                    ylim = 2*height - (2*height)/rad * x_pos
                    if np.round(y_pos - ylim, 4) <= 0:
                        slices[0].append(pos_idx)
                        slices[1].append(pos_idx+n_edge)
                    if np.round(y_pos - ylim, 4) >= 0:
                        slices[4].append(pos_idx)
                        slices[5].append(pos_idx+n_edge)
                if x_pos >= rad:
                    ylim = -2*height + (2*height)/rad * x_pos
                    if np.round(y_pos-ylim, 4) <= 0:
                        slices[2].append(pos_idx)
                        slices[3].append(pos_idx+n_edge)
                    if np.round(y_pos-ylim, 4) >= 0:
                        slices[4].append(pos_idx)
                        slices[5].append(pos_idx+n_edge)

            # first row
            else:
                if x_pos <= rad:
                    slices[0].append(pos_idx)
                    slices[1].append(pos_idx)
                if x_pos >= rad:
                    slices[2].append(pos_idx)
                    slices[3].append(pos_idx)

            pos_idx += 1  

    for i, this_slice in enumerate(slices):
        this_slice.append(center_pt_idx)
        slices[i] = np.unique(this_slice)

    edge_indices = np.setdiff1d(all_indices, center_indices)
    return positions, center_pt_idx, center_indices, edge_indices, vert_slice_dict, slices
