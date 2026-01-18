def sigmoid(x):
    return 1 / (1 + e ** (-x))


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
