import numpy as np
from matplotlib import pyplot as plt, tri

from source.mesh import Mesh


def plot_results(mesh: Mesh, magnitudes: np.ndarray, file_name: str = 'results.png'):
    triangulation = tri.Triangulation(
        x=mesh.coordinates2D[:, 0],
        y=mesh.coordinates2D[:, 1],
        triangles=mesh.nodes_of_elem
    )
    plt.tricontourf(triangulation, magnitudes)
    plt.colorbar()
    plt.savefig(file_name)
    plt.close()


def plot_displacements(mesh: Mesh, displacements: np.ndarray):
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


def plot_stress(mesh: Mesh, stress: np.ndarray, file_name: str = 'stress.png'):
    triangulation = tri.Triangulation(
        x=mesh.coordinates2D[:, 0],
        y=mesh.coordinates2D[:, 1],
        triangles=mesh.nodes_of_elem
    )
    plt.tripcolor(triangulation, stress)
    plt.colorbar()
    plt.savefig(file_name)
    plt.close()
