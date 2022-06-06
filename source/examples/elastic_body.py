from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

from source.mesh import Mesh
from matplotlib import tri

from source.fem.elasticity_setup import ElasticitySetup


def plot_results(mesh: Mesh, magnitudes: np.ndarray):
    triangulation = tri.Triangulation(
        x=mesh.coordinates2D[:, 0],
        y=mesh.coordinates2D[:, 1],
        triangles=mesh.nodes_of_elem
    )
    plt.tricontourf(triangulation, magnitudes)
    plt.colorbar()
    plt.savefig('results.png')
    plt.close()


def plot_dispalcements(mesh: Mesh, displacements: np.ndarray):
    before = tri.Triangulation(
        x=mesh.coordinates2D[:, 0],
        y=mesh.coordinates2D[:, 1],
        triangles=mesh.nodes_of_elem
    )
    plt.triplot(before, color='#1f77b4')
    after = tri.Triangulation(
        x=mesh.coordinates2D[:, 0] + displacements[:, 0],
        y=mesh.coordinates2D[:, 1] + displacements[:, 1],
        triangles=mesh.nodes_of_elem
    )
    plt.triplot(after, color='#ff7f0e')
    plt.grid()
    plt.savefig('displacements.png')
    plt.close()


if __name__ == '__main__':
    mesh = Mesh('meshes/nailed_board.msh')

    rhs_func = lambda x: np.array([0, 0])
    dirichlet_func = lambda x: np.array([0, 0])
    neumann_func = lambda x: np.array([-0.0, -0.5])


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
        young_modulus=MaterialProperty.AluminumAlloys.value[0],
        poisson_ratio=MaterialProperty.AluminumAlloys.value[1]
    )
    results = fem.solve()
    displacements = np.vstack((results[:mesh.nodes_num], results[mesh.nodes_num:])).T
    displacement_magnitudes = np.sqrt(displacements[:, 0] ** 2 + displacements[:, 1] ** 2)

    # plot results of FEM
    plot_results(mesh, displacement_magnitudes)
    plot_dispalcements(mesh, displacements)
