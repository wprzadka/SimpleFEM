import numpy as np
import pytest

import source.fem.utils as utils


@pytest.mark.parametrize('coords, area', [
    ([[0, 0], [0, 1], [1, 0]], 0.5),
    ([[-2, 3], [-3, -1], [3, -2]], 12.5)
])
def test_area_of_triangle(coords: np.ndarray, area: float):
    coords = np.array(coords)
    result = utils.area_of_triangle(coords)
    assert result == pytest.approx(area)
