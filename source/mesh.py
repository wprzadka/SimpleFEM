from enum import IntEnum

import numpy as np
import gmsh
import matplotlib.pyplot as plt
from matplotlib import tri


class Mesh:

    class ElementType(IntEnum):
        LINE = 1
        TRIANGULAR = 2

    def __init__(self, mesh_filename: str):
        gmsh.initialize()
        gmsh.open(mesh_filename)

        self.nodes_of_elem = None
        self.coordinates = None
        self.extract_triangular_elements()

        self.dirichlet_boundaries = None
        self.neumann_boundaries = None
        self.extract_boundary_conditions()

    def extract_triangular_elements(self):

        _, node_tags = gmsh.model.mesh.get_elements_by_type(self.ElementType.TRIANGULAR)
        node_tags = np.array(node_tags)

        # extract element to nodes mapping
        self.nodes_of_elem = np.array([tag - 1 for tag in node_tags]).reshape((-1, 3))
        # extract coordinates of nodes
        node_tags, coordinates, _ = gmsh.model.mesh.get_nodes_by_element_type(self.ElementType.TRIANGULAR)
        max_node_tag = gmsh.model.mesh.get_max_node_tag()
        self.coordinates = np.empty(shape=(max_node_tag, 3))
        self.coordinates[0, :] = coordinates[3 * 0: 3 * (0 + 1)]
        for i, tag in enumerate(node_tags):
            self.coordinates[int(tag - 1)] = coordinates[3 * i: 3 * (i+1)]

    def extract_boundary_conditions(self):
        elem_tags, node_tags = gmsh.model.mesh.get_elements_by_type(self.ElementType.LINE)
        self.dirichlet_boundaries = np.array([[int(beg - 1), int(end - 1)] for beg, end in node_tags.reshape((-1, 2))])
        # TODO extract neumann and dirichlet conditions
        self.neumann_boundaries = None

    def draw(self):
        triangulation = tri.Triangulation(
            x=self.coordinates[:, 0],
            y=self.coordinates[:, 1],
            triangles=self.nodes_of_elem
        )
        plt.triplot(triangulation)
        for boundaries, color in [
            (self.dirichlet_boundaries, (1, 0, 0, 0.5)),
            (self.neumann_boundaries, (0, 1, 0, 0.5))
        ]:
            if boundaries is None:
                continue
            boundary_coords = np.array([self.coordinates[idxs] for idxs in boundaries])
            for dx, dy in zip(boundary_coords[:, :, 0], boundary_coords[:, :, 1]):
                plt.plot(dx, dy, color=color)

        plt.savefig('mesh.png')
        plt.close()


if __name__ == '__main__':

    mesh = Mesh("meshes/rectangle.msh")
    mesh.draw()
