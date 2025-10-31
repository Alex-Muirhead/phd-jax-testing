from pathlib import Path
from gdtk import lmr
import numpy as np

REPO_ROOT = Path("/home/alex/GDTk/gdtk.robust-python-modules")
MAX_VERTICES = 4


def main():
    lmr_cfg = lmr.LmrConfig(REPO_ROOT / "src/lmr/lmr.cfg")
    sim_data = lmr.SimInfo(lmr_cfg)
    snap = sim_data.read_snapshot("0001")
    grid = snap.grids[0]

    # Build everything into a single array
    num_cells = grid.niv * grid.njv

    vertices = np.stack((grid.vertices.x, grid.vertices.y, grid.vertices.z), axis=-1)
    vertices = vertices.reshape((num_cells, 3))
    print(f"Vertices shape: {vertices.shape}")

    cells = np.zeros((num_cells, MAX_VERTICES), dtype=int)

    for j in range(grid.njv - 1):
        for i in range(grid.niv - 1):
            idx = j * grid.niv + i
            # Anti-clockwise, starting from bottom-left
            cells[idx, :] = [idx, idx + 1, idx + grid.niv + 1, idx + grid.niv]

    print(cells)


if __name__ == "__main__":
    main()
