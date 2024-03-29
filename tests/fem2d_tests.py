from typing import Callable

import numpy as np
import quadpy
import pytest

from source.utilities import computation_utils as utils
from source.fem.fem2d import FEM


class MeshMock:
    dirichlet_boundaries = np.array([[]])
    neumann_boundaries = np.array([[]])


@pytest.mark.parametrize('coords, func', [
    (np.array([[0, 0], [0, 1], [1, 0]]), lambda x: 1),
    (np.array([[0, 0], [0, 1], [1, 0]]), lambda x: x ** 3),
    (np.array([[9, 4], [2, 3], [5, 2]]), lambda x: x[0] ** 9 + x[1] ** 2 - 5),
    (np.array([[90, 0], [1, 1], [1, 2]]), lambda x: 1000 * x[0] * x[1]),
    (np.array([[100, 0], [10, 20], [50, 20]]), lambda x: np.exp(x[0]) + np.exp(x[1]))
])
def test_rhs(coords: np.ndarray, func: Callable[[np.ndarray], float]):
    wrap_func = lambda x: np.array([func(x) * utils.base_func(x, coords)])
    scheme = quadpy.t2.get_good_scheme(1)
    expected = np.squeeze(scheme.integrate(wrap_func, coords))
    fem = FEM(
        MeshMock(),
        func
    )

    result = fem.rhs_val(coords)

    assert result == pytest.approx(expected, rel=0.01)
