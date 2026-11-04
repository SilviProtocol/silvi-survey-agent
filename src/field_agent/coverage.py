"""Coverage grid computed from replayed (or live) poses.

The ground plane is the site-local XZ plane (Y is up). A cell counts
as seen when a pose came within ``seen_radius_m`` of its centre, or
when a recorded plant sits in it. The grid is sparse in memory and
encodes to the wire ``grid-v1`` form: row-major bytes (row = z
index), 0 = unseen, 1 = seen, base64.
"""

from __future__ import annotations

import base64
import math

UNSEEN = 0
SEEN = 1

DEFAULT_SEEN_RADIUS_M = 5.0


class CoverageGrid:
    def __init__(
        self,
        cell_m: float = 1.0,
        seen_radius_m: float = DEFAULT_SEEN_RADIUS_M,
    ):
        if cell_m <= 0:
            raise ValueError("cell_m must be positive")
        self.cell_m = float(cell_m)
        self.seen_radius_m = float(seen_radius_m)
        self._seen: set[tuple[int, int]] = set()
        self._min_i = self._min_j = self._max_i = self._max_j = None
        self._pose_min_i = self._pose_min_j = self._pose_max_i = self._pose_max_j = None

    # -- indexing ------------------------------------------------------------
    def cell_of(self, x: float, z: float) -> tuple[int, int]:
        return (math.floor(x / self.cell_m), math.floor(z / self.cell_m))

    def cell_center(self, i: int, j: int) -> tuple[float, float]:
        return ((i + 0.5) * self.cell_m, (j + 0.5) * self.cell_m)

    def _touch(self, i: int, j: int) -> None:
        if self._min_i is None:
            self._min_i = self._max_i = i
            self._min_j = self._max_j = j
        else:
            self._min_i = min(self._min_i, i)
            self._max_i = max(self._max_i, i)
            self._min_j = min(self._min_j, j)
            self._max_j = max(self._max_j, j)

    def _touch_pose(self, i: int, j: int) -> None:
        if self._pose_min_i is None:
            self._pose_min_i = self._pose_max_i = i
            self._pose_min_j = self._pose_max_j = j
        else:
            self._pose_min_i = min(self._pose_min_i, i)
            self._pose_max_i = max(self._pose_max_i, i)
            self._pose_min_j = min(self._pose_min_j, j)
            self._pose_max_j = max(self._pose_max_j, j)

    # -- updates ---------------------------------------------------------------
    def add_pose(self, position_m) -> None:
        """Mark cells within the seen radius of a pose."""
        x, z = float(position_m[0]), float(position_m[2])
        ci, cj = self.cell_of(x, z)
        reach = math.ceil(self.seen_radius_m / self.cell_m)
        for i in range(ci - reach, ci + reach + 1):
            for j in range(cj - reach, cj + reach + 1):
                cx, cz = self.cell_center(i, j)
                dist = math.hypot(cx - x, cz - z)
                if dist <= self.seen_radius_m:
                    self._seen.add((i, j))
                    self._touch(i, j)
                    self._touch_pose(i, j)

    def add_point(self, position_m) -> None:
        """Mark the single cell holding a point (e.g. a recorded plant)."""
        i, j = self.cell_of(float(position_m[0]), float(position_m[2]))
        self._seen.add((i, j))
        self._seen.add((i, j))
        self._touch(i, j)

    # -- queries -----------------------------------------------------------------
    @property
    def bounds(self) -> tuple[int, int, int, int] | None:
        """(min_i, min_j, max_i, max_j) inclusive, or None when empty."""
        if self._min_i is None:
            return None
        return (self._min_i, self._min_j, self._max_i, self._max_j)

    @property
    def pose_bounds(self) -> tuple[int, int, int, int] | None:
        """Bounds of pose-seen cells only: the ground actually walked near.

        Plant points can sit far beyond the walked area; gap planning
        without an explicit boundary uses this region, not ``bounds``.
        """
        if self._pose_min_i is None:
            return None
        return (self._pose_min_i, self._pose_min_j, self._pose_max_i, self._pose_max_j)

    def is_seen(self, i: int, j: int) -> bool:
        return (i, j) in self._seen

    def seen_cells(self) -> set[tuple[int, int]]:
        return set(self._seen)

    def _fraction(self, cells: set[tuple[int, int]]) -> float:
        b = self.pose_bounds or self.bounds
        if b is None:
            return 0.0
        total = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
        inside = sum(1 for i, j in cells if b[0] <= i <= b[2] and b[1] <= j <= b[3])
        return inside / total if total else 0.0

    def coverage_fraction(self) -> float:
        """Seen fraction of the walked (pose) bounds; 0 when nothing walked."""
        return self._fraction(self._seen)


    def to_wire(self) -> dict:
        b = self.bounds
        if b is None:
            return {
                "cell_m": self.cell_m,
                "encoding": "grid-v1",
                "cells_b64": "",
                "origin_xz_m": [0.0, 0.0],
                "width": 0,
                "height": 0,
                "note": "no poses yet",
            }
        min_i, min_j, max_i, max_j = b
        width = max_i - min_i + 1
        height = max_j - min_j + 1
        cells = bytearray(width * height)
        for i, j in self._seen:
            cells[(j - min_j) * width + (i - min_i)] = SEEN
        return {
            "cell_m": self.cell_m,
            "encoding": "grid-v1",
            "cells_b64": base64.b64encode(bytes(cells)).decode("ascii"),
            "origin_xz_m": [min_i * self.cell_m, min_j * self.cell_m],
            "width": width,
            "height": height,
            "note": None,
        }

    @classmethod
    def from_wire(cls, payload: dict) -> CoverageGrid:
        if payload.get("encoding") != "grid-v1":
            raise ValueError(f"cannot decode coverage encoding {payload.get('encoding')!r}")
        grid = cls(cell_m=float(payload["cell_m"]))
        width = int(payload.get("width") or 0)
        height = int(payload.get("height") or 0)
        if width == 0 or height == 0:
            return grid
        origin = payload.get("origin_xz_m") or [0.0, 0.0]
        min_i = round(float(origin[0]) / grid.cell_m)
        min_j = round(float(origin[1]) / grid.cell_m)
        raw = base64.b64decode(payload.get("cells_b64") or "")
        if len(raw) != width * height:
            raise ValueError("coverage cells_b64 length does not match width*height")
        for j in range(height):
            for i in range(width):
                if raw[j * width + i] == SEEN:
                    grid._seen.add((min_i + i, min_j + j))
                    grid._touch(min_i + i, min_j + j)
        return grid
