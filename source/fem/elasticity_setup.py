from typing import Optional

import numpy as np

from source.fem import utils
from source.fem.fem2d import FEM
from source.mesh import Mesh


class ElasticitySetup(FEM):

    def __init__(
            self,
            young_modulus,
            poisson_ratio,
            **kwargs
    ):
        super(ElasticitySetup, self).__init__(**kwargs)
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio
        self.lamb = self.young_modulus * self.poisson_ratio / (1 - self.poisson_ratio ** 2)
        # TODO check this
        self.mu = self.young_modulus / (1 - self.poisson_ratio)

    def stima3aux_symetric(self, coords: np.ndarray):
        # coords = np.array([['x1', 'y1'], ['x2', 'y2'], ['x3', 'y3']])
        T = utils.area_of_triangle(coords)

        # x_diff_mat[i, j] = (x[i+2] - x[i+1]) * (x[j+2] - x[j+1])
        x_diff = np.roll(coords[:, 0], 1) - np.roll(coords[:, 0], 2)
        x_diff_mat = np.expand_dims(x_diff, 1) @ np.expand_dims(x_diff, 0)

        # y_diff_mat[i, j] = (y[i+1] - y[i+2]) * (y[j+1] - y[j+2])
        y_diff = np.roll(coords[:, 1], -1) - np.roll(coords[:, 1], -2)
        y_diff_mat = np.expand_dims(y_diff, 1) @ np.expand_dims(y_diff, 0)

        return T, x_diff_mat, y_diff_mat

    def stima3aux_asymetric(self, coords: np.ndarray):
        # coords = np.array([['x1', 'y1'], ['x2', 'y2'], ['x3', 'y3']])
        T = utils.area_of_triangle(coords)

        # xy_diff_mat[i, j] = (x[i+2] - x[i+1]) * (x[j+2] - x[j+1])
        x_diff = np.roll(coords[:, 0], 1) - np.roll(coords[:, 0], 2)
        y_diff = np.roll(coords[:, 1], -1) - np.roll(coords[:, 1], -2)
        xy_diff_mat = np.expand_dims(x_diff, 1) @ np.expand_dims(y_diff, 0)

        return T, xy_diff_mat

    def stima3xx(self, coords: np.ndarray) -> np.ndarray:
        T, x_diff_mat, y_diff_mat = self.stima3aux_symetric(coords)
        return ((self.mu + self.lamb) * y_diff_mat + 0.5 * self.mu * x_diff_mat) / (4 * T)

    def stima3yy(self, coords: np.ndarray) -> np.ndarray:
        T, x_diff_mat, y_diff_mat = self.stima3aux_symetric(coords)
        return ((self.mu + self.lamb) * x_diff_mat + 0.5 * self.mu * y_diff_mat) / (4 * T)

    def stima3xy(self, coords: np.ndarray) -> np.ndarray:
        T, xy_diff_mat = self.stima3aux_asymetric(coords)
        return (self.lamb * xy_diff_mat + 0.5 * self.mu * xy_diff_mat.T) / (4 * T)

    def stima3yx(self, coords: np.ndarray) -> np.ndarray:
        T, xy_diff_mat = self.stima3aux_asymetric(coords)
        return (self.lamb * xy_diff_mat.T + 0.5 * self.mu * xy_diff_mat) / (4 * T)

    def solve(self):
        # we have 2 base functions for every node
        nodes_num = self.mesh.coordinates2D.shape[0]
        base_func_num = 2 * nodes_num
        A = np.zeros(shape=(base_func_num, base_func_num))
        b = np.zeros(shape=(base_func_num,))

        # Assembly stiffness matrix
        for func, beg_x, beg_y in [
            (self.stima3xx, 0, 0),
            (self.stima3xy, nodes_num, 0),
            (self.stima3yx, 0, nodes_num),
            (self.stima3yy, nodes_num, nodes_num)
        ]:
            for nodes in self.mesh.nodes_of_elem:
                local = func(self.mesh.coordinates2D[nodes])
                for y, col in enumerate(nodes):
                    for x, row in enumerate(nodes):
                        A[row + beg_x, col + beg_y] += local[x, y]
        # TODO optimize with Ayx = Axy.T

        # assembly rhs vector
        for nodes in self.mesh.nodes_of_elem:
            rhs_x, rhs_y = self.rhs_val(self.mesh.coordinates2D[nodes])
            b[nodes] += rhs_x
            b[nodes + nodes_num] += rhs_y

        # neumann conditions
        for vert_idxs in self.mesh.neumann_boundaries:
            neu_x, neu_y = self.neumann_val(self.mesh.coordinates2D[vert_idxs])
            b[vert_idxs] += neu_x
            b[vert_idxs + nodes_num] += neu_y

        # dirichlet conditions
        u = np.zeros(shape=(base_func_num, 1))
        for vert_idxs in self.mesh.dirichlet_boundaries:
            values = self.dirichlet_val(self.mesh.coordinates2D[vert_idxs])
            dir_x, dir_y = values[:, 0], values[:, 1]
            u[vert_idxs] = np.expand_dims(dir_x, 1)
            u[vert_idxs + nodes_num] = np.expand_dims(dir_y, 1)
        b -= (A @ u).T[0]

        free_nodes = [v for v in range(nodes_num) if v not in self.mesh.dirichlet_boundaries]
        free_nodes = free_nodes + [v + nodes_num for v in free_nodes]

        # solve
        u_free = np.linalg.solve(A[free_nodes][:, free_nodes], b[free_nodes])
        w = np.squeeze(u.copy(), 1)
        w[free_nodes] = u_free
        return w