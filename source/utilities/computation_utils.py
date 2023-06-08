import numpy as np


def area_of_triangle(coords: np.ndarray) -> float:
    # coords = np.array([['x1', 'y1'], ['x2', 'y2'], ['x3', 'y3']])
    doubleT = coords[1:3].T - np.expand_dims(coords[0], 1)
    areaT = np.abs(np.linalg.det(doubleT)) / 2
    return areaT


def center_of_mass(coords: np.ndarray) -> np.ndarray:
    return np.sum(coords, 0) / coords.shape[0]


def base_func(x: np.ndarray, vertices: np.ndarray):
    base = np.concatenate((x.T, vertices[0:2]), 0)
    numerator = np.concatenate((np.ones((3, 1)), base), 1)
    denominator = np.concatenate((np.ones((3, 1)), vertices), 1)
    return np.linalg.det(numerator) / np.linalg.det(denominator)

def interp_value(x: np.ndarray, verts: np.ndarray, values: np.ndarray):
    points = np.hstack((verts, values[:, None]))
    inner_vecs = points[1:] - points[0]
    normal = np.cross(inner_vecs[0], inner_vecs[1])
    val = values[0] - ((x - verts[0]) @ normal[:2]) / normal[2]
    return val

def grad_base_func(vertices: np.ndarray, idx: int):
    i = (idx + 1) % 3
    j = (idx + 2) % 3
    return np.array([vertices[i, 1] - vertices[j, 1], vertices[j, 0] - vertices[i, 0]]) \
           / (2 * area_of_triangle(vertices))
