from __future__ import division, print_function; __metaclass__ = type
import numpy as np

import MDAnalysis


def fib(N, N_tot):
    assert N <= N_tot

    if N == (N_tot+1)//2:
        return 0

    else:
        return (N-1)+fib(N-1, N_tot)


def gen_plate_position(n_mid=3, d=5):
    n_edge = fib(n_mid, n_mid)
    n_atoms = n_mid + 2*n_edge

    y_offset = np.sqrt(3)*0.5*d

    positions = np.zeros((n_atoms, 3))

    pos_idx = 0
    for i, n_row in enumerate(range(n_mid,(n_mid-1)//2,-1)):
        y_pos = i*y_offset

        for j in range(n_row):
            x_offset = i*0.5*d
            x_pos = x_offset + j*d

            positions[pos_idx] = x_pos, y_pos, 0.0
            if i > 0:
                positions[pos_idx+n_edge] = x_pos, -y_pos, 0.0
            pos_idx += 1

    return positions
