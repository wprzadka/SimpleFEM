from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

from source.mesh import Mesh
from matplotlib import tri

from source.fem.elasticity_setup import ElasticitySetup


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

    rhs_func = lambda x: np.array([0, 1])
    dirichlet_func = lambda x: np.array([0, 0])
    neumann_func = lambda x: np.array([0, 0])


    class MaterialProperty(Enum):
        AluminumAlloys = (10.2, 0.33)
        BerylliumCopper = (18.0, 0.29)
        CarbonSteel = (29.0, 0.29)
        CastIron = (14.5, 0.21)

    fem = ElasticitySetup(
        mesh=mesh,
        rhs_func=rhs_func,
        dirichlet_func=dirichlet_func,
        neumann_func=neumann_func,
        young_modulus=MaterialProperty.BerylliumCopper.value[0],
        poisson_ratio=MaterialProperty.BerylliumCopper.value[1]
    )
    results = fem.solve()
    displacements = results[:fem.mesh.nodes_num] + results[fem.mesh.nodes_num:]

    print("solution")
    print(displacements)

    # plot results of FEM
    plot_results(displacements)

    # validation
    exact = lambda x: -np.sin(x[0] * np.pi) * np.sin(x[1] * np.pi) / (2 * np.pi ** 2)
    plot_exact_solution(exact)
