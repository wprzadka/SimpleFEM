from typing import Optional
import numpy as np

from source.mesh import Mesh


class FEM:
    def __init__(
            self,
            mesh: Mesh,
            rhs_func: callable,
            dirichlet_func: Optional[callable],
            neumann_func: Optional[callable]
    ):
        self.mesh = mesh
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

    def solve(self):
        nodes_num = self.mesh.coordinates2D.shape[0]
        A = np.zeros(shape=(nodes_num, nodes_num))
        b = np.zeros(shape=(nodes_num,))

        # Assembly stiffness matrix
        for elem in self.mesh.nodes_of_elem:
            local = self.stima3(self.mesh.coordinates2D[elem])
            for y, col in enumerate(elem):
                for x, row in enumerate(elem):
                    A[row, col] += local[x, y]

        # assembly rhs vector
        for elem in self.mesh.nodes_of_elem:
            rhs = self.rhs_val(self.mesh.coordinates2D[elem])
            b[elem] += rhs

        # neumann conditions
        for vert_idxs in self.mesh.neumann_boundaries:
            b[vert_idxs] += self.neumann_val(self.mesh.coordinates2D[vert_idxs])

        # dirichlet conditions
        u = np.zeros(shape=(nodes_num, 1))
        for vert_idxs in self.mesh.dirichlet_boundaries:
            u[vert_idxs] = self.dirichlet_val(self.mesh.coordinates2D[vert_idxs])
        b -= (A @ u).T[0]

        free_nodes = [v for v in range(nodes_num) if v not in self.mesh.dirichlet_boundaries]

        # A = A[free_nodes][:, free_nodes]
        # b = b[free_nodes]

        # solve
        u_free = np.linalg.solve(A[free_nodes][:, free_nodes], b[free_nodes])
        w = np.squeeze(u.copy(), 1)
        w[free_nodes] = u_free
        return w