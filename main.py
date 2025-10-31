from pathlib import Path
from gdtk import lmr
import numpy as np

REPO_ROOT = Path("/home/alex/GDTk/gdtk.robust-python-modules")
MAX_VERTICES = 4
SENTINAL = -1


def main():
    lmr_cfg = lmr.LmrConfig(REPO_ROOT / "src/lmr/lmr.cfg")
    sim_data = lmr.SimInfo(lmr_cfg)
    snap = sim_data.read_snapshot("0001")
    grid = snap.grids[0]

    # Build everything into a single array
    nic = grid.niv - 1
    njc = grid.njv - 1
    num_cells = nic * njc
    num_verts = grid.niv * grid.njv

    vertices = np.stack((grid.vertices.x, grid.vertices.y, grid.vertices.z), axis=-1)
    vertices = vertices.reshape((num_verts, 3))
    print(f"Vertices shape: {vertices.shape}")

    cells = np.zeros((num_cells, MAX_VERTICES), dtype=int)
    neighbours = np.zeros((num_cells, MAX_VERTICES), dtype=int)

    # This is all trivial to setup for a structured grid
    for j in range(nic):
        for i in range(njc):
            cell_idx = j * nic + i
            # Anti-clockwise, starting from bottom-left
            cells[cell_idx, :] = [
                cell_idx,
                cell_idx + 1,
                cell_idx + grid.niv + 1,
                cell_idx + grid.niv,
            ]
            # Anti-clockwise, starting from 0rad (EAST)
            neighbours[cell_idx, :] = [
                cell_idx + 1 if i < nic else SENTINAL,
                cell_idx + nic if j < njc else SENTINAL,
                cell_idx - 1 if i > 0 else SENTINAL,
                cell_idx - nic if j > 0 else SENTINAL,
            ]

    print(f"Cell definition: shape = {cells.shape}")
    print(f"Cell neighbours: shape = {neighbours.shape}")


if __name__ == "__main__":
    main()
