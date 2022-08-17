from enum import Enum

import numpy as np


from source.fem.plotting_utils import plot_results, plot_displacements, plot_stress
from source.mesh import Mesh
from source.fem.elasticity_setup import ElasticitySetup


if __name__ == '__main__':
    # mesh = Mesh('meshes/nailed_board.msh')
    # print(mesh.physical_groups_mapping)
    # mesh.set_boundary_condition(Mesh.BoundaryConditionType.DIRICHLET, ['dirichlet:curves'])
    # mesh.set_boundary_condition(Mesh.BoundaryConditionType.NEUMANN, ['neumann:curves'])

    mesh = Mesh('meshes/bridge.msh')
    print(mesh.physical_groups_mapping)
    mesh.set_boundary_condition(Mesh.BoundaryConditionType.DIRICHLET, ['left_edge_bridge', 'right_edge_bridge'])
    mesh.set_boundary_condition(Mesh.BoundaryConditionType.NEUMANN, ['up_left_bridge'])

    rhs_func = lambda x: np.array([0, -0.2])
    dirichlet_func = lambda x: np.array([0, 0])
    neumann_func = lambda x: np.array([-0.0, -1.0])


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
        young_modulus=MaterialProperty.CarbonSteel.value[0],
        poisson_ratio=MaterialProperty.CarbonSteel.value[1]
    )
    results = fem.solve()
    displacements = np.vstack((results[:mesh.nodes_num], results[mesh.nodes_num:])).T
    displacement_magnitudes = np.sqrt(displacements[:, 0] ** 2 + displacements[:, 1] ** 2)

    # plot results of FEM
    plot_results(mesh, displacement_magnitudes)
    plot_displacements(mesh, displacements)

