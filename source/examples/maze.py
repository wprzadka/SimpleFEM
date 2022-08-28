from source.utilities.plotting_utils import plot_results
from source.mesh import Mesh
from source.fem.laplace_setup import LaplaceSetup


if __name__ == '__main__':

    mesh = Mesh('meshes/maze.msh')
    mesh.draw()

    rhs_func = lambda x: 100 * ((x[0] - 0.2) ** 2 + (x[1] - 0.2) ** 2)
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