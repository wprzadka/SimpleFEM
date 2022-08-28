from enum import Enum

import numpy as np


from source.fem.plotting_utils import plot_results, plot_displacements, plot_stress
from source.mesh import Mesh
from source.fem.elasticity_setup import ElasticitySetup


if __name__ == '__main__':
    mesh = Mesh('meshes/rectangle.msh')
    mesh.draw()
    print(mesh.physical_groups_mapping)
    mesh.set_boundary_condition(Mesh.BoundaryConditionType.DIRICHLET, ['left'])
    mesh.set_boundary_condition(Mesh.BoundaryConditionType.NEUMANN, ['right'])

    # mesh = Mesh('meshes/bridge.msh')
    # print(mesh.physical_groups_mapping)
    # mesh.set_boundary_condition(Mesh.BoundaryConditionType.DIRICHLET, ['left_edge_bridge', 'right_edge_bridge'])
    # mesh.set_boundary_condition(Mesh.BoundaryConditionType.NEUMANN, ['up_left_bridge'])

    rhs_func = lambda x: np.array([0, -9.81])
    dirichlet_func = lambda x: np.array([0, 0])
    neumann_func = lambda x: np.array([0, -9.81 * 1e2])  # weight of 100kg in earth gravity

    class MaterialProperty(Enum):
        CarbonSteel = (200e9, 0.28)
        Gold = (78e9, 0.44)
        Diamond = (1035e9, 0.29)  # Poisson's ratio = 0.1-0.29
        Polystyrene = (2.8e9, 0.38)  # Young's modulus = 2.8-3.5 GPa
        Silicon = (107e9, 0.27)
        Mica = (34.5e9, 0.205)

    fem = ElasticitySetup(
        mesh=mesh,
        rhs_func=rhs_func,
        dirichlet_func=dirichlet_func,
        neumann_func=neumann_func,
        young_modulus=MaterialProperty.Polystyrene.value[0],
        poisson_ratio=MaterialProperty.Polystyrene.value[1]
    )
    results = fem.solve()
    displacements = np.vstack((results[:mesh.nodes_num], results[mesh.nodes_num:])).T
    displacement_magnitudes = np.sqrt(displacements[:, 0] ** 2 + displacements[:, 1] ** 2)

    # plot results of FEM
    plot_results(mesh, displacement_magnitudes)
    zoom_factor = 1e4
    plot_displacements(mesh, displacements * zoom_factor)

