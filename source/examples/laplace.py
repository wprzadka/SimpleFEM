import matplotlib.pyplot as plt
import numpy as np

from source.mesh import Mesh
from matplotlib import tri

from source.fem.laplace_setup import LaplaceSetup


def plot_exact_solution(exact: callable):
    X, Y = np.meshgrid(np.linspace(0, 1), np.linspace(0, 1))
    exact_values = np.reshape(exact(np.stack((X, Y))), X.shape)
    plt.contour(X, Y, exact_values)
    plt.savefig('exact.png')
    plt.close()


def plot_results(results: np.ndarray):
    triangulation = tri.Triangulation(
        x=mesh.coordinates2D[:, 0],
        y=mesh.coordinates2D[:, 1],
        triangles=mesh.nodes_of_elem
    )
    plt.tricontourf(triangulation, results)
    plt.colorbar()
    plt.savefig('results.png')
    plt.close()


if __name__ == '__main__':

    mesh = Mesh('meshes/rectangle.msh')

    rhs_func = lambda x: -np.sin(x[0] * np.pi) * np.sin(x[1] * np.pi)
    # dirichlet_func = lambda x: 1 if x[0] < 0.1 or x[0] > 0.9 else 0
    dirichlet_func = lambda x: 0
    neumann_func = lambda x: 0

    fem = LaplaceSetup(
        mesh=mesh,
        rhs_func=rhs_func,
        dirichlet_func=dirichlet_func,
        neumann_func=neumann_func
    )
    results = fem.solve()
    print("solution")
    print(results)

    # plot results of FEM
    plot_results(results)

    # validation
    exact = lambda x: -np.sin(x[0] * np.pi) * np.sin(x[1] * np.pi) / (2 * np.pi ** 2)
    plot_exact_solution(exact)