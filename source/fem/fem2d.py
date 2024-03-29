from typing import Optional, Callable, Union
import numpy as np

from source.utilities import computation_utils as utils
from source.mesh import Mesh


class FEM:
    def __init__(
            self,
            mesh: Mesh,
            rhs_func: Callable[[np.ndarray], Union[float, np.ndarray]],
            dirichlet_func: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
            neumann_func: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None
    ):
        assert not (mesh.dirichlet_boundaries.size > 0 and dirichlet_func is None)
        assert not (mesh.neumann_boundaries.size > 0 and neumann_func is None)
        self.mesh = mesh
        self.rhs_func = rhs_func
        self.dirichlet_func = dirichlet_func
        self.neumann_func = neumann_func

    def rhs_val(self, vertices: np.ndarray) -> float:
        T = utils.area_of_triangle(vertices)
        center = utils.center_of_mass(vertices)
        return T / 3 * self.rhs_func(center)

    def neumann_val(self, vertices: np.array) -> float:
        # single vertex
        if vertices.shape[0] == 1:
            return self.neumann_func(vertices)
        # vertices = [[x0, y0], [x1, y1]]
        center = utils.center_of_mass(vertices)
        length = np.sqrt((vertices[1, 0] - vertices[0, 0]) ** 2 + (vertices[1, 1] - vertices[0, 1]) ** 2)

        return self.neumann_func(center) * length / 2

    def dirichlet_val(self, vertices: np.array) -> np.ndarray:
        # vertices = [[x0, y0], [x1, y1]]
        values = np.array([self.dirichlet_func(v) for v in vertices])
        return values

    def solve(self):
        raise NotImplementedError
