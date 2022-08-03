from enum import Enum
from typing import List, Dict

import numpy as np
import meshio
import matplotlib.pyplot as plt
from matplotlib import tri
import matplotlib.patches as mpatches


class Mesh:

    class ElementType(Enum):
        VERTEX = 'vertex'
        LINE = 'line'
        TRIANGULAR = 'triangle'

    class BoundaryConditionType(Enum):
        DIRICHLET = 'dirichlet'
        NEUMANN = 'neumann'

    def __init__(self, mesh_filename: str):

        msh = meshio.read(mesh_filename)

        # check if types of elements in mesh are supported
        elem_types = [t.value for t in self.ElementType]
        assert all([key in elem_types for key in msh.cells_dict.keys()])

        self.nodes_of_elem = msh.cells_dict[self.ElementType.TRIANGULAR.value]
        self.coordinates = msh.points
        self.coordinates2D = self.coordinates[:, 0:2]
        self.nodes_num = self.coordinates.shape[0]
        self.elems_num = len(self.nodes_of_elem)

        self.physical_groups_mapping = self.extract_physical_groups(msh)
        self.dirichlet_boundaries = np.array([])
        self.neumann_boundaries = np.array([])

    def extract_physical_groups(self, msh) -> Dict[str, np.ndarray]:
        condition_of_elem = self.prepare_physical_groups_mapping(msh)
        physical_groups_mapping = {}

        physical_groups = msh.cell_data['gmsh:physical']
        group_idx = 0
        inner_idx = 0
        for elem_type, elem_nodes in msh.cells_dict.items():

            for node in elem_nodes:
                key = physical_groups[group_idx][inner_idx]
                condition = condition_of_elem[key]
                if condition not in physical_groups_mapping:
                    physical_groups_mapping[condition] = []
                physical_groups_mapping[condition].append(node)

                inner_idx += 1
                if inner_idx >= physical_groups[group_idx].size:
                    inner_idx = 0
                    group_idx += 1

        return {k: np.array(v) for k, v in physical_groups_mapping.items()}

    def prepare_physical_groups_mapping(self, msh) -> list:
        condition_of_elem = ['undefined'] * (max([v[0] for v in msh.field_data.values()]) + 1)
        for group_name, val in msh.field_data.items():
            condition_of_elem[val[0]] = group_name
        return condition_of_elem

    def get_group_names(self):
        return self.physical_groups_mapping.keys()

    def set_boundary_condition(self, boundary_type: BoundaryConditionType, groups: List[str]):

        boundaries = np.concatenate(tuple(self.physical_groups_mapping[name] for name in groups))
        if boundary_type == Mesh.BoundaryConditionType.DIRICHLET:
            self.dirichlet_boundaries = boundaries
        elif boundary_type == Mesh.BoundaryConditionType.NEUMANN:
            self.neumann_boundaries = boundaries

    def draw(self, color_boundaries=False, color_palette=plt.cm.tab10):
        triangulation = tri.Triangulation(
            x=self.coordinates[:, 0],
            y=self.coordinates[:, 1],
            triangles=self.nodes_of_elem
        )
        plt.triplot(triangulation)

        if color_boundaries:
            for boundary, color in [
                (self.dirichlet_boundaries, (1, 0, 0, 0.5)),
                (self.neumann_boundaries, (0, 1, 0, 0.5))
            ]:
                if boundary.size == 0:
                    continue
                boundary_coords = np.array([self.coordinates[idxs] for idxs in boundary])
                for dx, dy in zip(boundary_coords[:, :, 0], boundary_coords[:, :, 1]):
                    plt.plot(dx, dy, color=color)
        else:
            patches = []
            for idx, (name, group) in enumerate(self.physical_groups_mapping.items()):
                if group.size == 0:
                    continue
                boundary_coords = np.array([self.coordinates[idxs] for idxs in group])
                for dx, dy in zip(boundary_coords[:, :, 0], boundary_coords[:, :, 1]):
                    plt.plot(dx, dy, color=color_palette(idx))
                patches.append(mpatches.Patch(color=color_palette(idx), label=name))
            plt.legend(handles=patches, loc='upper right')

        plt.savefig('mesh.png')
        plt.close()


if __name__ == '__main__':

    # mesh = Mesh("meshes/nailed_board.msh")
    # print(mesh.get_groups())
    # mesh.set_boundary_condition(Mesh.BoundaryConditionType.DIRICHLET, 'dirichlet:curves')
    # mesh.set_boundary_condition(Mesh.BoundaryConditionType.NEUMANN, 'neumann:curves')

    mesh = Mesh("meshes/bridge.msh")
    print(mesh.get_group_names())
    # mesh.set_boundary_condition(Mesh.BoundaryConditionType.DIRICHLET, ['left_edge_bridge', 'right_edge_bridge'])
    # mesh.set_boundary_condition(Mesh.BoundaryConditionType.NEUMANN, 'up_left_bridge')

    mesh.draw()
