import numpy as np
import matplotlib.pyplot as plt

from source.fem.plotting_utils import plot_results
from source.mesh import Mesh
from source.fem.laplace_setup import LaplaceSetup


def plot_exact_solution(exact: callable):
    X, Y = np.meshgrid(np.linspace(0, 1), np.linspace(0, 1))
    exact_values = np.reshape(exact(np.stack((X, Y))), X.shape)
    plt.contour(X, Y, exact_values)
    plt.savefig('exact.png')
    plt.close()


if __name__ == '__main__':

    mesh = Mesh('meshes/maze.msh')
    rhs_func = lambda x: 100 * ((x[0] - 0.2) ** 2 + (x[1] - 0.2) ** 2)
    # dirichlet_func = lambda x: 1 if x[0] < 0.1 or x[0] > 0.9 else 0
    # dirichlet_func = lambda x: 1
    neumann_func = lambda x: 100

    maze_edges = [v for v in mesh.get_group_names() if 'line' in v]
    mesh.set_boundary_condition(Mesh.BoundaryConditionType.NEUMANN, maze_edges)

    fem = LaplaceSetup(
        mesh=mesh,
        rhs_func=rhs_func,
        neumann_func=neumann_func
        # dirichlet_func=dirichlet_func
    )
    results = fem.solve()
    print("solution")
    print(results)

    # plot results of FEM
    plot_results(mesh, results)

    # validation
    # exact = lambda x: -np.sin(x[0] * np.pi) * np.sin(x[1] * np.pi) / (2 * np.pi ** 2)
    # plot_exact_solution(exact)
