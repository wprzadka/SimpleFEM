from typing import Optional
import numpy as np


class FEM:
    def __init__(
            self,
            rhs_func: callable,
            dirichlet_func: Optional[callable],
            neumann_func: Optional[callable]
    ):
        self.rhs_func = rhs_func
        self.dirichlet_func = dirichlet_func
        self.neumann_func = neumann_func

    def rhs_val(self, vertices: np.ndarray) -> float:
        T = vertices[1:3].T - np.expand_dims(vertices[0], 1)
        detT = np.linalg.det(T)
        center = np.sum(vertices, 0) / 3
        return detT / 6 * self.rhs_func(center)

    def neumann_val(self, vertices: np.array) -> float:
        # vertices = [[x0, y0], [x1, y1]]
        center = np.sum(vertices, 0) / 2
        length = np.sqrt((vertices[1, 0] - vertices[0, 0]) ** 2 + (vertices[1, 1] - vertices[0, 1]) ** 2)

        return self.neumann_func(center) * length / 2

    def dirichlet_val(self, vertices: np.array) -> np.ndarray:
        # vertices = [[x0, y0], [x1, y1]]
        values = np.array([self.dirichlet_func(v) for v in vertices])
        return np.expand_dims(values, 1)
