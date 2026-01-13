import numpy as np
import random
import matplotlib.pyplot as plt
from backwards_euler import insert_matrix

def sigmoid(x):
    return 1/(1 + e**(-x))

def poly(coefficients, x):
    return sum(coefficient * x**n for n, coefficient in enumerate(coefficients))

def poly_map(points):
    degree_x = random.randint(1, 10)
    coefficients_x = [np.random.randint(-points, points) for _ in range(0, degree_x+1)]
    degree_y = random.randint(1, 10)
    coefficients_y = [np.random.randint(-points, points) for _ in range(0, degree_y+1)]
    hmap = np.array([[abs(poly(coefficients_x, x/(points*degree_x)) + poly(coefficients_y, y/(points*degree_y))) for x in range(0, points)] for y in range(0, points)], dtype=float)
    return hmap

def piecewise_poly_map(points):
    midpoint = int(points/2)
    hmap = np.zeros((points, points))
    tl = poly_map(midpoint)
    hmap = insert_matrix(tl, hmap, 0, 0)
    tr = poly_map(midpoint)
    hmap = insert_matrix(tr, hmap, 0, midpoint)
    bl = poly_map(midpoint)
    hmap = insert_matrix(bl, hmap, midpoint, 0)
    insert_matrix(tl, hmap, 0, 0)
    br = poly_map(midpoint)
    hmap = insert_matrix(tl, hmap, midpoint, midpoint)
    return hmap

def main():
    points = 50
    print(piecewise_poly_map(50))
    hmap = piecewise_poly_map(points)
    fig, axis = plt.subplots()
    pcm = axis.pcolormesh(hmap, cmap=plt.cm.jet, vmin=0, vmax=hmap.max())
    plt.colorbar(pcm, ax=axis)
    pcm.set_array(hmap)
    plt.show()

if __name__ == "__main__":
    main()
