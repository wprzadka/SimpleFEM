import numpy as np
import pytest

import source.fem.utils as utils


@pytest.mark.parametrize('coords, area', [
    (np.array([[0, 0], [0, 1], [1, 0]]), 0.5),
    (np.array([[-2, 3], [-3, -1], [3, -2]]), 12.5)
])
def test_area_of_triangle(coords: np.ndarray, area: float):
    result = utils.area_of_triangle(coords)
    assert result == pytest.approx(area)


@pytest.mark.parametrize('coords', [
    np.array([0, 1]),
    np.array([2, 100]),
    np.array([[0, 0], [0, 1], [1, 0]]),
    np.array([[-2, 3], [-3, -1], [3, -2]]),
    np.array([[-2, 3, 1], [-3, -1, 2], [3, -2, 2], [4, 2, 6]])
])
def test_center_of_mass(coords: np.ndarray):
    center = np.zeros(coords[0].shape)
    points_counter = 0
    for point in coords:
        center += point
        points_counter += 1
    center /= points_counter

    result = utils.center_of_mass(coords)

    assert result == pytest.approx(center)
