"""Deterministic gap planner.

Interface: ``plan_gaps(grid, boundary_xz=None, *, min_cells=2) -> list[Gap]``.

Plain code, no model. Unseen cells inside the boundary (or the grid bounds
when no boundary is given) are flood-filled into 4-connected clusters;
clusters smaller than ``min_cells`` are dropped. Results are ordered by
size descending, then by centre, so the same grid always yields the same
gaps in the same order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from field_agent.coverage import CoverageGrid


@dataclass(frozen=True)
class Gap:
    center_xz_m: tuple[float, float]
    radius_m: float
    n_cells: int
    area_m2: float

    def center_xyz_m(self, y_m: float = 0.0) -> list[float]:
        return [self.center_xz_m[0], y_m, self.center_xz_m[1]]


def _point_in_polygon(x: float, z: float, polygon_xz) -> bool:
    """Even-odd ray cast in the XZ plane."""
    inside = False
    n = len(polygon_xz)
    for a in range(n):
        x1, z1 = polygon_xz[a]
        x2, z2 = polygon_xz[(a + 1) % n]
        if (z1 > z) != (z2 > z):
            x_cross = x1 + (z - z1) * (x2 - x1) / (z2 - z1)
            if x < x_cross:
                inside = not inside
    return inside


def plan_gaps(
    grid: CoverageGrid,
    boundary_xz=None,
    *,
    min_cells: int = 2,
) -> list[Gap]:
    # Without a boundary, plan inside the walked (pose) region: distant
    # plant points must not turn the whole outside into one giant "gap".
    bounds = grid.bounds if boundary_xz else (grid.pose_bounds or grid.bounds)
    if bounds is None:
        return []
    min_i, min_j, max_i, max_j = bounds
    polygon = None
    if boundary_xz:
        polygon = [(float(p[0]), float(p[1])) for p in boundary_xz]
        if len(polygon) < 3:
            raise ValueError("boundary needs at least three points")

    def is_gap_cell(i: int, j: int) -> bool:
        if grid.is_seen(i, j):
            return False
        if polygon is not None:
            cx, cz = grid.cell_center(i, j)
            return _point_in_polygon(cx, cz, polygon)
        return True

    gaps: list[Gap] = []
    visited: set[tuple[int, int]] = set()
    for j in range(min_j, max_j + 1):
        for i in range(min_i, max_i + 1):
            if (i, j) in visited or not is_gap_cell(i, j):
                continue
            stack = [(i, j)]
            visited.add((i, j))
            cluster: list[tuple[int, int]] = []
            while stack:
                ci, cj = stack.pop()
                cluster.append((ci, cj))
                for ni, nj in ((ci + 1, cj), (ci - 1, cj), (ci, cj + 1), (ci, cj - 1)):
                    if (
                        min_i <= ni <= max_i
                        and min_j <= nj <= max_j
                        and (ni, nj) not in visited
                        and is_gap_cell(ni, nj)
                    ):
                        visited.add((ni, nj))
                        stack.append((ni, nj))
            if len(cluster) < min_cells:
                continue
            centers = [grid.cell_center(ci, cj) for ci, cj in cluster]
            cx = sum(c[0] for c in centers) / len(centers)
            cz = sum(c[1] for c in centers) / len(centers)
            area = len(cluster) * grid.cell_m * grid.cell_m
            gaps.append(
                Gap(
                    center_xz_m=(round(cx, 3), round(cz, 3)),
                    radius_m=round(math.sqrt(area / math.pi), 3),
                    n_cells=len(cluster),
                    area_m2=round(area, 3),
                )
            )
    gaps.sort(key=lambda g: (-g.n_cells, g.center_xz_m[0], g.center_xz_m[1]))
    return gaps
