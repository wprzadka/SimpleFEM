import numpy as np


def area_of_triangle(coords: np.ndarray) -> float:
    # coords = np.array([['x1', 'y1'], ['x2', 'y2'], ['x3', 'y3']])
    doubleT = coords[1:3].T - np.expand_dims(coords[0], 1)
    areaT = np.abs(np.linalg.det(doubleT)) / 2
    return areaT


def center_of_mass(coords: np.ndarray) -> np.ndarray:
    return np.sum(coords, 0) / coords.shape[0]
