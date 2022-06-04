import matplotlib.pyplot as plt
import numpy as np

from mesh import Mesh
from matplotlib import tri

from fem.laplace_setup import LaplaceSetup


if __name__ == '__main__':
    mesh = Mesh('meshes/rectangle.msh')

    rhs_func = lambda x: -np.sin(x[0] * np.pi) * np.sin(x[1] * np.pi)
    # dirichlet_func = lambda x: 1 if x[0] < 0.1 or x[0] > 0.9 else 0
    dirichlet_func = lambda x: 0
    neumann_func = lambda x: 0

    fem = LaplaceSetup(
        rhs_func=rhs_func,
        dirichlet_func=dirichlet_func,
        neumann_func=neumann_func
    )

    nodes_num = mesh.coordinates2D.shape[0]
    A = np.zeros(shape=(nodes_num, nodes_num))
    b = np.zeros(shape=(nodes_num,))

    # Assembly stiffness matrix
    for elem in mesh.nodes_of_elem:
        local = fem.stima3(mesh.coordinates2D[elem])
        for y, col in enumerate(elem):
            for x, row in enumerate(elem):
                A[row, col] += local[x, y]

    # assembly rhs vector
    for elem in mesh.nodes_of_elem:
        rhs = fem.rhs_val(mesh.coordinates2D[elem])
        b[elem] += rhs

    # neumann conditions
    for vert_idxs in mesh.neumann_boundaries:
        b[vert_idxs] += fem.neumann_val(mesh.coordinates2D[vert_idxs])

    # dirichlet conditions
    u = np.zeros(shape=(nodes_num, 1))
    for vert_idxs in mesh.dirichlet_boundaries:
        u[vert_idxs] = fem.dirichlet_val(mesh.coordinates2D[vert_idxs])
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
