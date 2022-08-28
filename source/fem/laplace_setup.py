import numpy as np

from source.utilities import computation_utils as utils
from source.fem.fem2d import FEM


class LaplaceSetup(FEM):

    def stima3(self, vertices: np.ndarray):
        # cords = np.array([['x1', 'y1'], ['x2', 'y2'], ['x3', 'y3']])
        T = utils.area_of_triangle(vertices)
        G_b = np.vstack((
            np.ones(3),
            vertices[:, 0],
            vertices[:, 1]
        ))
        G = np.linalg.inv(G_b) @ np.array([[0, 0], [1, 0], [0, 1]])
        M = (T / 2) * (G @ G.T)
        return M

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
            values = self.dirichlet_val(self.mesh.coordinates2D[vert_idxs])
            u[vert_idxs] = np.expand_dims(values, 1)
        b -= (A @ u).T[0]

        free_nodes = [v for v in range(nodes_num) if v not in self.mesh.dirichlet_boundaries]

        # A = A[free_nodes][:, free_nodes]
        # b = b[free_nodes]

        # solve
        u_free = np.linalg.solve(A[free_nodes][:, free_nodes], b[free_nodes])
        w = np.squeeze(u.copy(), 1)
        w[free_nodes] = u_free
        return w
