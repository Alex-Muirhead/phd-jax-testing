from collections import namedtuple
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import math

import numpy as np
from gdtk import lmr

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Self
    from gdtk.geom.sgrid import StructuredGrid

REPO_ROOT = Path("/home/alex/GDTk/gdtk.robust-python-modules")
MAX_VERTICES = 4
SENTINAL = -1


def construct_halfspace(vertices: np.ndarray) -> np.ndarray: ...


@dataclass
class Grid:
    geometry: GridGeometry
    topology: GridTopology

    @classmethod
    def from_structured_grid(cls, sgrid: StructuredGrid) -> Self:
        # We assume the structured grid will always be HEX cells

        Shape = namedtuple("Shape", "i, j, k")
        vert_grid_shape = Shape(i=sgrid.niv, j=sgrid.njv, k=sgrid.nkv)
        cell_grid_shape = Shape(i=sgrid.niv - 1, j=sgrid.njv - 1, k=max(sgrid.nkv - 1, 1))

        num_verts = math.prod(vert_grid_shape)
        num_cells = math.prod(cell_grid_shape)

        match sgrid.dimensions:
            case 2:
                facets_per_cell = 4
                vertices_per_cell = 4
            case 3:
                facets_per_cell = 6
                vertices_per_cell = 8
            case _:
                raise ValueError("Grid must have 2 or 3 dimensions")

        vertex_coordinates = np.stack(
            (sgrid.vertices.x, sgrid.vertices.y, sgrid.vertices.z), axis=-1
        ).reshape((num_verts, 3))

        # EAST, WEST, NORTH, SOUTH
        neighbour_offsets = np.array([[+1, 0, 0], [-1, 0, 0], [0, +1, 0], [0, -1, 0]])
        # Anti-clockwise from BOTTOM-LEFT
        vertex_offsets = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])

        cell_vertices = np.full((num_cells, vertices_per_cell), fill_value=SENTINAL)
        cell_adjacency = np.full((num_cells, facets_per_cell), fill_value=SENTINAL)

        # We include the dual-edges to the "exterior cell"
        num_dual_edges = 2 * num_cells + cell_grid_shape.i + cell_grid_shape.j
        dual_edges = np.full((num_dual_edges, 2), fill_value=SENTINAL)
        dual_edge_lookup = dict()

        cell_connection_list = np.full((num_cells, facets_per_cell), fill_value=SENTINAL)

        edge_counter = 0
        for cell_idx in range(num_cells):
            (i, j, k) = np.unravel_index(cell_idx, cell_grid_shape)

            vertices = vertex_offsets + (i, j, k)
            vertex_idxs = np.ravel_multi_index(vertices.T, vert_grid_shape)
            cell_vertices[cell_idx, :] = vertex_idxs

            neighbours = neighbour_offsets + (i, j, k)
            valid = np.all(
                np.logical_and(neighbours >= 0, neighbours < cell_grid_shape),
                axis=1,
            )
            neighbour_idxs = np.ravel_multi_index(neighbours.T, cell_grid_shape, mode="wrap")
            # We can remove this if we want to have "wrapped" domain
            neighbour_idxs[~valid] = SENTINAL
            cell_adjacency[cell_idx, :] = neighbour_idxs

            for local_facet_id, neighbour in enumerate(neighbour_idxs):
                dual_edge = (cell_idx, int(neighbour))
                # Check if the (flipped) dual-edge already exists first
                edge_idx = dual_edge_lookup.get(dual_edge[::-1], edge_counter)
                cell_connection_list[cell_idx, local_facet_id] = edge_idx

                if edge_idx != edge_counter:
                    # The edge did already exist
                    continue

                dual_edges[edge_idx] = dual_edge
                dual_edge_lookup[dual_edge] = edge_idx
                edge_counter += 1

        # Ugh I wish we had chained operators...
        # TODO: Extend to 3D grids
        i, j, k = np.meshgrid(*map(range, vert_grid_shape), sparse=True)

        # Oh GOD this is awful...
        exterior_cell_vertex_list = []
        slicer = (slice(None, None, None),) * 3
        for dim_idx in range(sgrid.dimensions):
            exterior_slice = list(slicer)
            exterior_slice[dim_idx] = 0
            exterior_slice = tuple(exterior_slice)
            exterior_cell_vertex_list.append(
                np.ravel_multi_index(
                    (i[exterior_slice], j[exterior_slice], k[exterior_slice]),
                    vert_grid_shape,
                    mode="wrap",
                ).flatten()
            )
            exterior_slice = list(slicer)
            exterior_slice[dim_idx] = -1
            exterior_slice = tuple(exterior_slice)
            exterior_cell_vertex_list.append(
                np.ravel_multi_index(
                    (i[exterior_slice], j[exterior_slice], k[exterior_slice]),
                    vert_grid_shape,
                    mode="wrap",
                ).flatten()
            )

        exterior_cell_vertices = np.hstack(exterior_cell_vertex_list)
        exterior_cell_vertices = np.unique(exterior_cell_vertices)
        print(exterior_cell_vertices)

        for facet_id, (cell_a, cell_b) in enumerate(dual_edges):
            if cell_a == SENTINAL and cell_b == SENTINAL:
                raise ValueError(f"Invalid edge at idx: {facet_id}")

            vertices_a = cell_vertices[cell_a, :] if cell_a != SENTINAL else exterior_cell_vertices
            vertices_b = cell_vertices[cell_b, :] if cell_b != SENTINAL else exterior_cell_vertices
            # The vertex order doesn't actually matter!
            # We only need to find the normal vector (and maybe tangent ones)
            facet_vertices = np.intersect1d(vertices_a, vertices_b)
            print(facet_vertices)

        print("Vertex shape: ", vert_grid_shape)
        # geometry = GridGeometry(vertex_coordinates=vertex_coordinates, cell_half_spaces=)
        # topology = GridTopology(cell_vertices=cell_vertices, cell_adjacency=cell_adjacency)

        # return Grid(geometry=geometry, topology=topology)


@dataclass
class GridGeometry:
    # For v vertices, e edges, f faces, and c cells
    # For ~n facets per face (d+1 if using simplices)
    vertex_coordinates: np.ndarray  # Coordinates (v,d)
    facet_hyperplanes: np.ndarray  # Normal vector & offset -> (f,d+1)


@dataclass
class GridTopology:
    # For v vertices, e edges, and f faces
    # For ~n vertices per face (d+1 if using simplices)
    cell_facets: np.ndarray  # Facet-ids
    facet_vertices: np.ndarray  # Vertex-ids
    cell_adjacency: np.ndarray  # Face-to-face connections (f,~n)


def main():
    lmr_cfg = lmr.LmrConfig(REPO_ROOT / "src/lmr/lmr.cfg")
    sim_data = lmr.SimInfo(lmr_cfg)
    snap = sim_data.read_snapshot("0001")
    grid = snap.grids[0]

    Grid.from_structured_grid(grid)


if __name__ == "__main__":
    main()
