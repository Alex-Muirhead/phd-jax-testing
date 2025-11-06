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
                euler_characteristic = 2
            case 3:
                facets_per_cell = 6
                euler_characteristic = 0
            case _:
                raise ValueError("Grid must have 2 or 3 dimensions")

        # So this is fun...
        #   X-coord increases along axis=0
        #   Y-coord increases along axis=1
        # Let's transpose axis 0 and 1, so that we get "ij" indexing

        vertex_coordinates = (
            np.stack((sgrid.vertices.x, sgrid.vertices.y, sgrid.vertices.z), axis=-1)
            .transpose((1, 0, 2))  # We move from "xy" matrix indexing to "ij" (see numpy.meshgrid)
            .reshape((num_verts, 3))
        )

        # EAST, WEST, NORTH, SOUTH
        neighbour_offsets = np.array([[+1, 0, 0], [-1, 0, 0], [0, +1, 0], [0, -1, 0]])
        neighbour_idxs = np.full((num_cells, facets_per_cell), fill_value=SENTINAL)

        num_dual_edges = 2 * num_cells - cell_grid_shape.i - cell_grid_shape.j
        dual_edges = np.full((num_dual_edges, 2), fill_value=SENTINAL)
        dual_edge_lookup = dict()

        edge_idx = 0
        for cell_idx in range(num_cells):
            (i, j, k) = np.unravel_index(cell_idx, cell_grid_shape)
            neighbours = neighbour_offsets + (i, j, k)
            valid = np.all(
                np.logical_and(neighbours >= 0, neighbours < cell_grid_shape),
                axis=1,
            )
            cell_neighbour_ids = np.ravel_multi_index(neighbours.T, cell_grid_shape, mode="wrap")
            # We can remove this if we want to have "wrapped" domain
            cell_neighbour_ids[~valid] = SENTINAL
            neighbour_idxs[cell_idx, :] = cell_neighbour_ids

            # Build our edge-list
            for neighbour_idx in cell_neighbour_ids[valid]:
                dual_edge = (cell_idx, int(neighbour_idx))
                if tuple(reversed(dual_edge)) in dual_edge_lookup:
                    continue
                dual_edge_lookup[dual_edge] = edge_idx
                dual_edges[edge_idx, :] = dual_edge
                edge_idx += 1

        print(edge_idx, num_dual_edges)
        print(dual_edges)


@dataclass
class GridGeometry:
    # For v vertices, e edges, and f faces
    vertex_coordinates: np.ndarray  # Coordinates (v,d)
    facet_hyperplanes: np.ndarray  # Normal Vector & Tangent Vectors (e,d,d)


@dataclass
class GridTopology:
    # For v vertices, e edges, and f faces
    # For ~n vertices per face (d+1 if using simplices)
    cell_facets: np.ndarray  # Face vertices (f,~n)
    dual_edges: np.ndarray  # Face-to-face connections (e,2)


def main():
    lmr_cfg = lmr.LmrConfig(REPO_ROOT / "src/lmr/lmr.cfg")
    sim_data = lmr.SimInfo(lmr_cfg)
    snap = sim_data.read_snapshot("0001")
    grid = snap.grids[0]

    Grid.from_structured_grid(grid)


if __name__ == "__main__":
    main()
