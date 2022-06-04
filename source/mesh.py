import numpy as np
import gmsh
import matplotlib.pyplot as plt
from matplotlib import tri


class Mesh:

    def __init__(self, mesh_filename: str):
        gmsh.initialize()
        gmsh.open(mesh_filename)

        TRIANGULAR_TYPE = 2
        elem_tags, node_tags = gmsh.model.mesh.get_elements_by_type(TRIANGULAR_TYPE)

        self.elem_tags = np.array(elem_tags)
        self.node_tags = np.array(node_tags)

        # max_elem_tag = gmsh.model.mesh.get_max_element_tag()
        # elem_tag_to_idx = np.full(shape=max_elem_tag + 1, fill_value=-1)
        # for idx, tag in enumerate(np.unique(self.elem_tags)):
        #     elem_tag_to_idx[tag] = idx

        max_node_tag = gmsh.model.mesh.get_max_node_tag()
        node_tag_to_idx = np.full(shape=max_node_tag + 1, fill_value=-1)
        for idx, tag in enumerate(np.unique(self.node_tags)):
            node_tag_to_idx[tag] = idx

        # extract element to nodes mapping
        self.nodes_of_elem = np.array([node_tag_to_idx[tag] for tag in self.node_tags]).reshape((-1, 3))
        # extract coordinates of nodes
        node_tags, coordinates, _ = gmsh.model.mesh.get_nodes_by_element_type(TRIANGULAR_TYPE)
        self.coordinates = np.empty(shape=(max_node_tag, 3))
        for i, tag in enumerate(node_tags):
            self.coordinates[node_tag_to_idx[tag]] = coordinates[3 * i: 3 * (i+1)]

        for i in range(1, 4):
            print(gmsh.model.mesh.get_element_properties(i))

    def draw(self):
        triangulation = tri.Triangulation(
            x=self.coordinates[:, 0],
            y=self.coordinates[:, 1],
            triangles=self.nodes_of_elem
        )
        plt.triplot(triangulation)
        plt.savefig('mesh.png')
        plt.close()


if __name__ == '__main__':

    mesh = Mesh("meshes/rectangle.msh")
    mesh.draw()
