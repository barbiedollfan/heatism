import numpy as np
import random
import matplotlib.pyplot as plt
from backwards_euler import insert_matrix

def sigmoid(x):
    return 1/(1 + e**(-x))

def linear_norm(old_min, old_max, new_min, new_max, x):
    if old_max == old_min:
        return x
    return (new_max - new_min) / (old_max - old_min) * (x - old_min) + new_min

def normalize(hmap, new_min, new_max):
    n = len(hmap)
    old_min = hmap.min()
    old_max = hmap.max()
    for x in range(0, n):
        for y in range(0, n):
            hmap[x, y] = linear_norm(old_min, old_max, new_min, new_max, hmap[x, y])
    return hmap

def poly(coefficients, x):
    return sum(coefficient * x**n for n, coefficient in enumerate(coefficients))

def poly_map(points, new_min, new_max):
    degree_x = random.randint(0, points)
    coefficients_x = [np.random.randint(-points, points) for _ in range(0, degree_x+1)]
    degree_y = random.randint(0, points)
    coefficients_y = [np.random.randint(-points, points) for _ in range(0, degree_y+1)]
    hmap = np.array([[poly(coefficients_x, x) + poly(coefficients_y, y) for x in range(0, points)] for y in range(0, points)], dtype=float)
    return normalize(hmap, new_min, new_max)

def constant_map(points, new_min, new_max):
    return np.full((points, points), np.random.randint(new_min, new_max))

def piecewise_poly_map(points, new_min, new_max):
    midpoint = int(points/2)
    hmap = np.zeros((points, points))
    tl = poly_map(midpoint, new_min, new_max)
    hmap = insert_matrix(tl, hmap, 0, 0)
    tr = poly_map(midpoint, new_min, new_max)
    hmap = insert_matrix(tr, hmap, 0, midpoint)
    bl = poly_map(midpoint, new_min, new_max)
    hmap = insert_matrix(bl, hmap, midpoint, 0)
    br = poly_map(midpoint, new_min, new_max)
    hmap = insert_matrix(br, hmap, midpoint, midpoint)
    return hmap

def piecewise_map(points, new_min, new_max):
    midpoint = int(points/2)
    hmap = np.zeros((points, points))
    tl = constant_map(midpoint, new_min, new_max)
    hmap = insert_matrix(tl, hmap, 0, 0)
    tr = constant_map(midpoint, new_min, new_max)
    hmap = insert_matrix(tr, hmap, 0, midpoint)
    bl = constant_map(midpoint, new_min, new_max)
    hmap = insert_matrix(bl, hmap, midpoint, 0)
    br = constant_map(midpoint, new_min, new_max)
    hmap = insert_matrix(br, hmap, midpoint, midpoint)
    return hmap

def main():
    points = 100
    hmap = piecewise_poly_map(points, 273, 1000)
    fig, axis = plt.subplots()
    pcm = axis.pcolormesh(hmap, cmap=plt.cm.jet, vmin=hmap.min(), vmax=hmap.max())
    plt.colorbar(pcm, ax=axis)
    pcm.set_array(hmap)
    plt.show()

if __name__ == "__main__":
    main()
