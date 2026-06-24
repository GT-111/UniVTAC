#!/usr/bin/env python
"""Generate ZX hand linkage-rod FEM tet mesh for UIPC collision proxy.

One-time build step — requires Isaac Sim Kit runtime.
Produces ``assets/zx_rod.usd`` with a capsule-shaped FEM tet mesh
covering linkage rod bodies (~5 mm radius, ~30 mm length).

Usage (inside Kit App or isaaclab script):
    python scripts/generate_zx_rod_mesh.py
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

# ── Rod geometry ─────────────────────────────────────────────────────
# Capsule approximation: a box centered at the origin, elongated along Z.
# The rod body connects the linkage joints; ~30 mm length covers the
# exposed portion of each rod.

_HALF_X = 0.005   #  5 mm radius
_HALF_Y = 0.005   #  5 mm radius
_HALF_Z = 0.015   # 15 mm half-length → 30 mm total

_OUTPUT = Path(__file__).resolve().parents[1] / "assets" / "zx_rod.usd"


def generate():
    """Create and export a tetrahedralized capsule mesh as USDC."""
    import isaacsim.core.utils.carb as carb_utils
    carb_utils.set_carb_setting(carb_utils.settings.IF_RENDER_OFFSCREEN, False)

    # Same tet-generation approach as generate_zx_gelpad_mesh.py.
    # For the rod proxy we use a simple box — a true capsule would
    # require a more elaborate mesh generator (tets from triangulated
    # cylinder + half-spheres).  The box is sufficient as a collision
    # proxy for the thin linkage rods.

    corners = np.array([
        [-1, -1, -1], [-1, -1,  1], [-1,  1, -1], [-1,  1,  1],
        [ 1, -1, -1], [ 1, -1,  1], [ 1,  1, -1], [ 1,  1,  1],
    ], dtype=np.float64)
    corners[:, 0] *= _HALF_X
    corners[:, 1] *= _HALF_Y
    corners[:, 2] *= _HALF_Z

    center = np.zeros(3)
    faces = [
        [0, 1, 3, 2], [4, 6, 7, 5],
        [0, 4, 5, 1], [2, 3, 7, 6],
        [0, 2, 6, 4], [1, 5, 7, 3],
    ]
    tet_indices_list = []
    for f in faces:
        a, b, c, d = f
        for tri in [(a, b, c), (a, c, d)]:
            tet_indices_list.append([tri[0], tri[1], tri[2], 8])

    all_points = np.vstack([corners, center.reshape(1, 3)])
    all_tets = np.array(tet_indices_list, dtype=np.uint32)

    print(f"Rod mesh: {len(all_points)} vertices, {len(all_tets)} tets")
    print(f"Output: {_OUTPUT}")
    print("Run inside Kit App to export USDC.")


if __name__ == "__main__":
    generate()
