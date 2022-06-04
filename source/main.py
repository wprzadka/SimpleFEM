import matplotlib.pyplot as plt
import numpy as np

from mesh import Mesh
from matplotlib import tri


def stima3(vertices: np.ndarray):
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


def b_val(vertices: np.ndarray, func: callable) -> float:
    T = vertices[1:3].T - np.expand_dims(vertices[0], 1)
    detT = np.linalg.det(T)
    center = np.sum(vertices, 0) / 3
    return detT / 6 * func(center)


def neumann_val(vertices: np.array, func: callable) -> float:
    # vertices = [[x0, y0], [x1, y1]]
    center = np.sum(vertices, 0) / 2
    length = np.sqrt((vertices[1, 0] - vertices[0, 0]) ** 2 + (vertices[1, 1] - vertices[0, 1]) ** 2)

    return func(center) * length / 2


def dirichlet_val(vertices: np.array, func: callable) -> np.ndarray:
    # vertices = [[x0, y0], [x1, y1]]
    values = np.array([func(v) for v in vertices])
    return np.expand_dims(values, 1)


if __name__ == '__main__':
    mesh = Mesh('meshes/rectangle.msh')

    func = lambda x: -np.sin(x[0] * np.pi) * np.sin(x[1] * np.pi)
    # dirichlet_func = lambda x: 1 if x[0] < 0.1 or x[0] > 0.9 else 0
    dirichlet_func = lambda x: 0
    neumann_func = lambda x: 0

    nodes_num = mesh.coordinates2D.shape[0]
    A = np.zeros(shape=(nodes_num, nodes_num))
    b = np.zeros(shape=(nodes_num,))

    # Assembly stiffness matrix
    for elem in mesh.nodes_of_elem:
        local = stima3(mesh.coordinates2D[elem])
        for y, col in enumerate(elem):
            for x, row in enumerate(elem):
                A[row, col] += local[x, y]

    # assembly rhs vector
    for elem in mesh.nodes_of_elem:
        rhs = b_val(mesh.coordinates2D[elem], func)
        b[elem] += rhs

    # neumann conditions
    for vert_idxs in mesh.neumann_boundaries:
        b[vert_idxs] += neumann_val(mesh.coordinates2D[vert_idxs], neumann_func)

    # dirichlet conditions
    u = np.zeros(shape=(nodes_num, 1))
    for vert_idxs in mesh.dirichlet_boundaries:
        u[vert_idxs] = dirichlet_val(mesh.coordinates2D[vert_idxs], dirichlet_func)
    b -= (A @ u).T[0]

    free_nodes = [v for v in range(nodes_num) if v not in mesh.dirichlet_boundaries]

    # A = A[free_nodes][:, free_nodes]
    # b = b[free_nodes]

    # solve
    u_free = np.linalg.solve(A[free_nodes][:, free_nodes], b[free_nodes])
    w = np.squeeze(u.copy(), 1)
    w[free_nodes] = u_free

    print("solution")
    print(w)

    # plot results of FEM
    triangulation = tri.Triangulation(
        x=mesh.coordinates2D[:, 0],
        y=mesh.coordinates2D[:, 1],
        triangles=mesh.nodes_of_elem
    )
    plt.tricontourf(triangulation, w)
    plt.colorbar()
    plt.savefig('results.png')
    plt.close()

    # validation
    exact = lambda x: -np.sin(x[0] * np.pi) * np.sin(x[1] * np.pi) / (2 * np.pi ** 2)

    X, Y = np.meshgrid(np.linspace(0, 1), np.linspace(0, 1))
    exact_values = np.reshape(exact(np.stack((X, Y))), X.shape)
    # plt.close()
    plt.contour(X, Y, exact_values)
    plt.savefig('exact.png')
