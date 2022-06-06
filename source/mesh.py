from enum import Enum
from typing import Tuple

import numpy as np
import meshio
import matplotlib.pyplot as plt
from matplotlib import tri
from numpy import ndarray


class Mesh:

    class ElementType(Enum):
        VERTEX = 'vertex'
        LINE = 'line'
        TRIANGULAR = 'triangle'

    elem_to_condition_type = {
        'vertex': 'points',
        'line': 'curves',
        'triangle': 'surfaces'
    }

    def __init__(self, mesh_filename: str):

        msh = meshio.read(mesh_filename)

        # check if types of elements in mesh are supported
        elem_types = [t.value for t in self.ElementType]
        assert all([key in elem_types for key in msh.cells_dict.keys()])

        self.nodes_of_elem = msh.cells_dict[self.ElementType.TRIANGULAR.value]
        self.coordinates = msh.points
        self.coordinates2D = self.coordinates[:, 0:2]
        self.nodes_num = self.coordinates.shape[0]

        dirichlet, neumann = self.extract_boundary_conditions(msh)
        self.dirichlet_boundaries = dirichlet
        self.neumann_boundaries = neumann

    def extract_boundary_conditions(self, msh) -> Tuple[ndarray, ndarray]:
        dirichlet = []
        neumann = []
        condition_of_elem = self.prepare_physical_groups_mapping(msh)

        physical_groups = msh.cell_data['gmsh:physical']
        group_idx = 0
        inner_idx = 0
        for elem_type, elem_nodes in msh.cells_dict.items():
            group_conditions = condition_of_elem[self.elem_to_condition_type[elem_type]]
            for node in elem_nodes:
                if 'dirichlet' in group_conditions \
                        and physical_groups[group_idx][inner_idx] in group_conditions['dirichlet']:
                    dirichlet.append(node)
                if 'neumann' in group_conditions \
                        and physical_groups[group_idx][inner_idx] in group_conditions['neumann']:
                    neumann.append(node)

                inner_idx += 1
                if inner_idx >= physical_groups[group_idx].size:
                    inner_idx = 0
                    group_idx += 1

        return np.array(dirichlet), np.array(neumann)

    def prepare_physical_groups_mapping(self, msh) -> dict:
        condition_of_elem = {}
        for key, val in msh.field_data.items():
            condition_type, element_type = key.split(':')
            if element_type not in condition_of_elem:
                condition_of_elem[element_type] = {}
            if condition_type not in condition_of_elem[element_type]:
                condition_of_elem[element_type][condition_type] = []
            condition_of_elem[element_type][condition_type] += list(val)
        return condition_of_elem

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
            if boundaries.size == 0:
                continue
            boundary_coords = np.array([self.coordinates[idxs] for idxs in boundaries])
            for dx, dy in zip(boundary_coords[:, :, 0], boundary_coords[:, :, 1]):
                plt.plot(dx, dy, color=color)

        plt.savefig('mesh.png')
        plt.close()


if __name__ == '__main__':

    mesh = Mesh("meshes/nailed_board.msh")
    mesh.draw()
