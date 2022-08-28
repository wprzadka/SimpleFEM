import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri

from source.utilities.plotting_utils import plot_results
from source.mesh import Mesh
from source.fem.laplace_setup import LaplaceSetup


def plot_exact_solution(exact: callable):
    X, Y = np.meshgrid(np.linspace(0, 1), np.linspace(0, 1))
    exact_values = np.reshape(exact(np.stack((X, Y))), X.shape)
    plt.contourf(X, Y, exact_values)
    plt.colorbar()
    plt.savefig('exact.png')
    plt.close()


def plot_error(mesh: Mesh, exact: callable, results: np.ndarray, file_name: str = 'errors.png'):
    triangulation = tri.Triangulation(
        x=mesh.coordinates2D[:, 0],
        y=mesh.coordinates2D[:, 1],
        triangles=mesh.nodes_of_elem
    )
    exact_val = np.array([exact(mesh.coordinates2D[node]) for node in range(mesh.nodes_num)])
    error = results - exact_val
    plt.tricontourf(triangulation, error)
    plt.colorbar()
    plt.savefig(file_name)
    plt.close()


if __name__ == '__main__':

    mesh = Mesh('meshes/square.msh')
    mesh.draw()

    rhs_func = lambda x: -2 * np.pi ** 2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    dirichlet_func = lambda x: 0

    mesh.set_boundary_condition(Mesh.BoundaryConditionType.DIRICHLET, ['right', 'down', 'left', 'up'])

    fem = LaplaceSetup(
        mesh=mesh,
        rhs_func=rhs_func,
        # neumann_func=neumann_func
        dirichlet_func=dirichlet_func
    )
    results = fem.solve()
    print("solution")
    print(results)

    # plot results of FEM
    plot_results(mesh, results)

    # validation
    exact = lambda x: np.sin(x[0] * np.pi) * np.sin(x[1] * np.pi)
    plot_exact_solution(exact)
    plot_error(mesh=mesh, exact=exact, results=results)
