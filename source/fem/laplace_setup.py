import numpy as np
from source.fem.fem2d import FEM


class LaplaceSetup(FEM):

    def stima3(self, vertices: np.ndarray):
        # cords = np.array([['x1', 'y1'], ['x2', 'y2'], ['x3', 'y3']])
        doubleT = vertices[1:3].T - np.expand_dims(vertices[0], 1)
        detT = np.linalg.det(doubleT) / 2
        G_b = np.vstack((
            np.ones(3),
            vertices[:, 0],
            vertices[:, 1]
        ))
        G = np.linalg.inv(G_b) @ np.array([[0, 0], [1, 0], [0, 1]])
        M = (detT / 2) * (G @ G.T)
        return M
