import numpy as np
import scipy.sparse as spr
import scipy.sparse.linalg as spl

def insert_matrix(small_matrix, large_matrix, c, r):
    s_rows, s_cols = small_matrix.shape

    large_matrix[r:r + s_rows, c:c + s_cols] = small_matrix
    return large_matrix

def gen_coeff_matrix(n, diag, hor):
    N = n**2 # For an original n by n grid of interior points, the coefficient matrix becomes n^2 by n^2
    A = np.zeros((N, N))
    for y in range(0, n):
        for x in range(0, n):
            k = x + n * y
            A[k, k] = diag
            if x > 0:
                A[k-1, k] = hor
            if x < n - 1:
                A[k+1, k] = hor
            if y > 0:
                A[k, k-n] = hor
            if y < n - 1:
                A[k, k+n] = hor
    return A

def gen_known_vector(prev_temps, r):
    inner_points = prev_temps[1:-1, 1:-1]
    inner_points[0, :] += r * prev_temps[0, 1:-1]
    inner_points[-1, :] += r * prev_temps[-1, 1:-1]
    inner_points[:, 0] += r * prev_temps[1:-1, 0]
    inner_points[:, -1] += r * prev_temps[1:-1, -1]
    return inner_points.flatten()

def next_temps(solve_matrix, prev_temps, r):
    n = len(prev_temps)
    t = gen_known_vector(prev_temps, r)
    new_temps_vec = solve_matrix(t) 
    new_temps_matrix = new_temps_vec.reshape(n-2, n-2)
    prev_temps = insert_matrix(new_temps_matrix, prev_temps, 1, 1)
    return prev_temps
