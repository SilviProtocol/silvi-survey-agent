"""Survey world: everything the agent knows about the survey, from wire v1.

Fed only by wire messages (poses, plant sightings, plant snapshots,
coverage) plus server-side results set through ``set_species`` / notes /
the walked boundary. All geometry is plain code here, never the model's
job: poses are camera-to-world and the camera looks down -Z.
"""

from __future__ import annotations

import itertools
import math
import re
from typing import ClassVar

from field_agent.revisit import rotate_vector

_ID_RE = re.compile(r"^\s*(?:t(?:ree)?|plant)?\s*-?\s*(\d+)\s*$", re.IGNORECASE)

NEED_ORDER = ["unconfirmed", "no_species"]

# Coverage: a ground cell counts as seen when a pose came within this
# range of its centre; coverage is judged only on cells within
# JUDGE_RADIUS_M of the walked path, so far-away plant points cannot
# inflate the unseen ground.
CELL_M = 1.0
SEEN_RADIUS_M = 5.0
JUDGE_RADIUS_M = 6.0


def parse_tree_id(text) -> int | None:
    """Accept 'T14', 'tree 14', 'plant 14' and '14'; anything else -> None."""
    match = _ID_RE.match(str(text))
    return int(match.group(1)) if match else None


def tree_label(plant_id: int) -> str:
    return f"T{plant_id}"


def _forward_xz(quaternion_xyzw) -> tuple[float, float]:
    fx, _fy, fz = rotate_vector(quaternion_xyzw, (0.0, 0.0, -1.0))
    length = math.hypot(fx, fz)
    if length < 1e-9:
        return (0.0, -1.0)
    return (fx / length, fz / length)


def relative_angle_deg(position_m, quaternion_xyzw, point_m) -> float:
    """Signed angle from the camera's ground heading to a point.

    Positive means the point is to the walker's right.
    """
    fx, fz = _forward_xz(quaternion_xyzw)
    rx, rz = -fz, fx  # right = (-forward_z, forward_x)
    dx = float(point_m[0]) - float(position_m[0])
    dz = float(point_m[2]) - float(position_m[2])
    return math.degrees(math.atan2(rx * dx + rz * dz, fx * dx + fz * dz))


def clock_of(angle_deg: float) -> int:
    clock = round(angle_deg / 30.0) % 12
    return 12 if clock == 0 else clock


def where_of(angle_deg: float) -> str:
    a = abs(angle_deg)
    if a <= 25.0:
        return "ahead"
    if a >= 135.0:
        return "behind"
    return "right" if angle_deg > 0 else "left"






def point_in_polygon(x: float, z: float, polygon_xz: list[tuple[float, float]]) -> bool:
    inside = False
    n = len(polygon_xz)
    for i in range(n):
        x1, z1 = polygon_xz[i]
        x2, z2 = polygon_xz[(i + 1) % n]
        if (z1 > z) != (z2 > z):
            cross_x = x1 + (z - z1) * (x2 - x1) / (z2 - z1)
            if x < cross_x:
                inside = not inside
    return inside


def _point_segment_distance(px, pz, ax, az, bx, bz) -> float:
    dx, dz = bx - ax, bz - az
    length2 = dx * dx + dz * dz
    if length2 <= 0:
        return math.hypot(px - ax, pz - az)
    t = max(0.0, min(1.0, ((px - ax) * dx + (pz - az) * dz) / length2))
    return math.hypot(px - (ax + t * dx), pz - (az + t * dz))


def _douglas_peucker(points: list[tuple[float, float]], tol: float) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return list(points)
    ax, az = points[0]
    bx, bz = points[-1]
    worst, worst_i = -1.0, 0
    for i in range(1, len(points) - 1):
        d = _point_segment_distance(points[i][0], points[i][1], ax, az, bx, bz)
        if d > worst:
            worst, worst_i = d, i
    if worst <= tol:
        return [points[0], points[-1]]
    left = _douglas_peucker(points[: worst_i + 1], tol)
    right = _douglas_peucker(points[worst_i:], tol)
    return left[:-1] + right


def simplify_closed_loop(points_xz: list[tuple[float, float]], tol: float = 0.3):
    """Simplify a walked loop: split at the point farthest from the start
    and Douglas-Peucker each half (plain DP collapses a closed loop)."""
    pts = [tuple(p) for p in points_xz]
    if len(pts) < 3:
        return pts
    sx, sz = pts[0]
    far = max(range(len(pts)), key=lambda i: math.hypot(pts[i][0] - sx, pts[i][1] - sz))
    if far == 0:
        return _douglas_peucker(pts, tol)
    first = _douglas_peucker(pts[: far + 1], tol)
    second = _douglas_peucker(pts[far:] + [pts[0]], tol)
    return first[:-1] + second[:-1]


class SurveyWorld:
    def __init__(self):
        self.poses: dict[int, dict] = {}
        self._pose_order: list[int] = []
        self.sightings: dict[int, set[int]] = {}
        self.snapshots: dict[int, dict] = {}
        self.coverage_payload: dict | None = None
        self.species: dict[int, dict] = {}
        self.notes: dict[int, list[str]] = {}
        self.waypoints: list[dict] = []
        self.tracking_losses = 0
        self._was_tracking = True
        self.boundary_state = "idle"  # idle | recording | paused | closed
        self.boundary_points: list[list[float]] = []
        self.boundary_polygon: list[list[float]] | None = None
        # Wire v2: the surveyor's manual pins, the phone's current page,
        # and the numbers the phone draws on screen.
        self.anchors: dict[str, dict] = {}
        self.mode: str | None = None
        self.display_numbers: dict[int, int] = {}

    # -- wire feed -----------------------------------------------------------
    def add_pose(self, payload: dict, *, t_mono_ms: float = 0.0) -> None:
        seq = int(payload["keyframe_seq"])
        tracking = str(payload.get("tracking") or "unknown")
        is_tracking = tracking == "tracking"
        if self._was_tracking and not is_tracking:
            self.tracking_losses += 1
        self._was_tracking = is_tracking
        if seq not in self.poses:
            self._pose_order.append(seq)
        self.poses[seq] = {
            "position_m": [float(x) for x in payload["position_m"]],
            "quaternion_xyzw": [float(x) for x in payload["quaternion_xyzw"]],
            "tracking": tracking,
            "t_mono_ms": float(t_mono_ms),
        }
        if self.boundary_state == "recording" and is_tracking:
            self.boundary_points.append([float(x) for x in payload["position_m"]])

    def add_sighting(self, payload: dict) -> None:
        plant_id = payload.get("plant_id")
        if plant_id is None:
            return
        self.sightings.setdefault(int(plant_id), set()).add(int(payload["keyframe_seq"]))

    def add_snapshot(self, payload: dict) -> None:
        self.snapshots[int(payload["plant_id"])] = dict(payload)
        if payload.get("display_no") is not None:
            self.display_numbers[int(payload["plant_id"])] = int(payload["display_no"])

    def add_coverage(self, payload: dict) -> None:
        self.coverage_payload = dict(payload)

    def add_anchor(self, payload: dict, *, t_mono_ms: float = 0.0) -> None:
        """One manual pin from the phone (wire v2 ``anchor``).

        The receive time (walk clock) rides along: a pin is only
        "where the walker just pointed" for a short while.
        """
        anchor_id = str(payload.get("anchor_id") or "")
        if not anchor_id:
            return
        if payload.get("action") == "removed":
            self.anchors.pop(anchor_id, None)
            return
        entry = dict(payload)
        entry["_received_mono_ms"] = float(t_mono_ms)
        self.anchors[anchor_id] = entry

    def set_display_no(self, plant_id: int, display_no: int) -> None:
        self.display_numbers[int(plant_id)] = int(display_no)

    def display_of(self, live_id: int) -> int | None:
        """The number the phone draws for this live plant.

        A merge keeps the lower number: when plants merge into this
        one, the smallest number among them is what stays on screen.
        """
        numbers = [
            self.display_numbers[pid]
            for pid in (live_id, *self.also_known_as(live_id))
            if pid in self.display_numbers
        ]
        return min(numbers) if numbers else None

    # -- server-side results ---------------------------------------------------
    def set_species(self, plant_id: int, species: dict) -> None:
        self.species[int(plant_id)] = dict(species)

    def add_note(self, plant_id: int, text: str) -> None:
        self.notes.setdefault(int(plant_id), []).append(str(text))

    # -- pose helpers -----------------------------------------------------------
    @property
    def current_pose(self) -> dict | None:
        if not self._pose_order:
            return None
        return self.poses[self._pose_order[-1]]

    def walking_speed_mps(self, window_s: float = 5.0) -> float:
        if len(self._pose_order) < 2:
            return 0.0
        latest = self.poses[self._pose_order[-1]]
        cutoff = latest["t_mono_ms"] - window_s * 1000.0
        recent = [
            self.poses[s] for s in self._pose_order if self.poses[s]["t_mono_ms"] >= cutoff
        ]
        if len(recent) < 2:
            return 0.0
        span_s = (recent[-1]["t_mono_ms"] - recent[0]["t_mono_ms"]) / 1000.0
        if span_s <= 0:
            return 0.0
        dist = 0.0
        for a, b in itertools.pairwise(recent):
            dist += math.hypot(
                b["position_m"][0] - a["position_m"][0],
                b["position_m"][2] - a["position_m"][2],
            )
        return dist / span_s

    def camera_pitch_deg(self) -> float | None:
        pose = self.current_pose
        if pose is None:
            return None
        _fx, fy, _fz = rotate_vector(pose["quaternion_xyzw"], (0.0, 0.0, -1.0))
        return round(math.degrees(math.asin(max(-1.0, min(1.0, fy)))), 0)

    # -- identity resolution -----------------------------------------------------
    def _merged_target(self, plant_id: int) -> int | None:
        snap = self.snapshots.get(plant_id)
        if snap is None:
            return None
        if snap.get("merged_into") is not None:
            return int(snap["merged_into"])
        return None

    def resolve(self, plant_id: int) -> tuple[str, int]:
        """-> (state, live_id): state is live | merged | retired | unknown.

        For merged, live_id is the tree the ID was merged into (chains
        followed). Retired trees stay retired even if a chain existed.
        """
        snap = self.snapshots.get(plant_id)
        if snap is None and plant_id not in self.sightings:
            return ("unknown", plant_id)
        if snap is not None and snap.get("status") == "retired":
            return ("retired", plant_id)
        current, seen = plant_id, set()
        while current not in seen:
            seen.add(current)
            target = self._merged_target(current)
            if target is None:
                break
            current = target
        if current != plant_id:
            return ("merged", current)
        if snap is not None and snap.get("status") == "merged":
            return ("merged", current)
        return ("live", current)

    def also_known_as(self, live_id: int) -> list[int]:
        known = []
        for pid in self.snapshots:
            if pid == live_id:
                continue
            state, resolved = self.resolve(pid)
            if state == "merged" and resolved == live_id:
                known.append(pid)
        return sorted(known)

    def live_ids(self) -> list[int]:
        ids = []
        for pid in self.snapshots:
            state, live = self.resolve(pid)
            if state in ("live", "merged") and live == pid:
                ids.append(pid)
        return sorted(ids)

    # -- per-tree facts -------------------------------------------------------------
    def _sighting_seqs(self, live_id: int) -> set[int]:
        seqs: set[int] = set()
        for pid in [live_id, *self.also_known_as(live_id)]:
            seqs |= self.sightings.get(pid, set())
        return seqs

    def views(self, live_id: int) -> int:
        seqs = self._sighting_seqs(live_id)
        if seqs:
            return len(seqs)
        snap = self.snapshots.get(live_id) or {}
        return int(snap.get("n_views") or 0)

    def needs(self, live_id: int) -> list[str]:
        snap = self.snapshots.get(live_id) or {}
        found = []
        if snap.get("status") != "confirmed":
            found.append("unconfirmed")
        if live_id not in self.species:
            found.append("no_species")
        return [n for n in NEED_ORDER if n in found]

    def is_complete(self, live_id: int) -> bool:
        snap = self.snapshots.get(live_id) or {}
        return snap.get("status") == "confirmed"

    def tree_row(self, live_id: int) -> dict | None:
        snap = self.snapshots.get(live_id)
        if snap is None:
            return None
        position = snap.get("position_m")
        row: dict = {
            "id": tree_label(live_id),
            "status": snap.get("status"),
            "position_m": [float(x) for x in position] if position else None,
            "sigma_m": snap.get("sigma_m"),
            "views": self.views(live_id),
            "also_known_as": [tree_label(p) for p in self.also_known_as(live_id)],
            "species": self._species_row(live_id),
            "needs": self.needs(live_id),
            "evidence_needs": self._evidence_needs(live_id),
            "notes": list(self.notes.get(live_id, [])),
        }
        pose = self.current_pose
        if pose is not None and position:
            angle = relative_angle_deg(pose["position_m"], pose["quaternion_xyzw"], position)
            dist = math.hypot(
                position[0] - pose["position_m"][0], position[2] - pose["position_m"][2]
            )
            row.update(
                dist_m=round(dist, 1),
                clock=clock_of(angle),
                where=where_of(angle),
                angle_deg=round(angle, 1),
            )
        else:
            row.update(dist_m=None, clock=None, where=None)
        return row

    _EVIDENCE_NEED_NAMES: ClassVar[dict[str, str]] = {
        "bark": "bark_closeup",
        "leaf": "leaf_closeup",
        "flower_or_fruit": "flower_or_fruit",
        "whole": "whole",
    }

    def _evidence_needs(self, live_id: int) -> list[dict]:
        """What the species plug wants next for this tree, best first."""
        result = self.species.get(live_id) or {}
        out = []
        for item in result.get("next_evidence") or []:
            want = str(item.get("want")) if isinstance(item, dict) else str(item)
            if want in self._EVIDENCE_NEED_NAMES:
                out.append({"need": self._EVIDENCE_NEED_NAMES[want], "want": want})
        return out

    # -- phone status rows ----------------------------------------------------
    def what_ring(self, live_id: int) -> float:
        result = self.species.get(live_id) or {}
        conf = result.get("confidence")
        return round(float(conf), 3) if conf is not None else 0.0

    def short_name(self, live_id: int) -> str:
        """The label under a plant's rings: at most 18 characters."""
        result = self.species.get(live_id) or {}
        name = result.get("common_name") or result.get("taxon") or ""
        if result.get("rank") == "genus" and name:
            genus = str(result.get("taxon") or name).split()[0]
            name = f"{genus} sp."
        return str(name)[:18]

    def status_row(self, live_id: int) -> dict:
        return {
            "plant_id": int(live_id),
            "number": self.display_of(live_id),
            "what_ring": self.what_ring(live_id),
            "short_name": self.short_name(live_id),
        }

    def _species_row(self, live_id: int) -> dict | None:
        result = self.species.get(live_id)
        if not result:
            return None
        row = {
            "taxon": result.get("taxon"),
            "rank": result.get("rank") or "species",
            "confidence": result.get("confidence"),
        }
        if result.get("runner_up"):
            row["runner_up"] = result["runner_up"]
        # Plug state: what evidence would help, and whether
        # the species side considers this plant settled.
        if result.get("common_name"):
            row["common_name"] = result["common_name"]
        if result.get("next_evidence"):
            row["next_evidence"] = [dict(n) for n in result["next_evidence"]]
        if result.get("done") is not None:
            row["species_done"] = bool(result["done"])
        return row

    def focus_id(self) -> int | None:
        pose = self.current_pose
        if pose is None:
            return None
        best, best_score = None, None
        for live_id in self.live_ids():
            snap = self.snapshots.get(live_id) or {}
            position = snap.get("position_m")
            if not position:
                continue
            angle = relative_angle_deg(pose["position_m"], pose["quaternion_xyzw"], position)
            dist = math.hypot(
                position[0] - pose["position_m"][0], position[2] - pose["position_m"][2]
            )
            if abs(angle) <= 15.0 and dist <= 8.0:
                score = abs(angle) / 15.0 + dist / 8.0
                if best_score is None or score < best_score:
                    best, best_score = live_id, score
        return best

    # -- scene / coverage -----------------------------------------------------------
    def scene_rows(self, radius_m: float = 12.0) -> list[dict]:
        rows = []
        for live_id in self.live_ids():
            row = self.tree_row(live_id)
            if row is None or row["dist_m"] is None or row["dist_m"] > radius_m:
                continue
            rows.append(row)
        rows.sort(key=lambda r: (r["where"] != "ahead", r["dist_m"], r["id"]))
        return rows

    def _judged_cells(self) -> set[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        reach = math.ceil(JUDGE_RADIUS_M / CELL_M)
        for pose in self.poses.values():
            if pose["tracking"] != "tracking":
                continue
            ci = math.floor(pose["position_m"][0] / CELL_M)
            cj = math.floor(pose["position_m"][2] / CELL_M)
            for i in range(ci - reach, ci + reach + 1):
                for j in range(cj - reach, cj + reach + 1):
                    cx, cz = (i + 0.5) * CELL_M, (j + 0.5) * CELL_M
                    if math.hypot(cx - pose["position_m"][0], cz - pose["position_m"][2]) <= JUDGE_RADIUS_M:
                        cells.add((i, j))
        return cells

    def _seen_cells(self) -> set[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        reach = math.ceil(SEEN_RADIUS_M / CELL_M)
        for pose in self.poses.values():
            if pose["tracking"] != "tracking":
                continue
            ci = math.floor(pose["position_m"][0] / CELL_M)
            cj = math.floor(pose["position_m"][2] / CELL_M)
            for i in range(ci - reach, ci + reach + 1):
                for j in range(cj - reach, cj + reach + 1):
                    cx, cz = (i + 0.5) * CELL_M, (j + 0.5) * CELL_M
                    if math.hypot(cx - pose["position_m"][0], cz - pose["position_m"][2]) <= SEEN_RADIUS_M:
                        cells.add((i, j))
        return cells

    def coverage_fraction(self) -> float:
        judged = self._judged_cells()
        if not judged:
            return 0.0
        return len(judged & self._seen_cells()) / len(judged)

    def nearest_unseen_ground(self) -> dict | None:
        pose = self.current_pose
        if pose is None:
            return None
        unseen = self._judged_cells() - self._seen_cells()
        if not unseen:
            return None
        best = None
        for i, j in unseen:
            cx, cz = (i + 0.5) * CELL_M, (j + 0.5) * CELL_M
            dist = math.hypot(cx - pose["position_m"][0], cz - pose["position_m"][2])
            if best is None or dist < best[0]:
                best = (dist, cx, cz)
        dist, cx, cz = best
        y = pose["position_m"][1]
        angle = relative_angle_deg(
            pose["position_m"], pose["quaternion_xyzw"], [cx, y, cz]
        )
        return {"dist_m": round(dist, 1), "clock": clock_of(angle), "position_m": [cx, y, cz]}

    # -- boundary ---------------------------------------------------------------------
    def boundary_action(self, action: str) -> dict:
        action = str(action).strip().lower()
        if action == "start":
            if self.boundary_state != "recording":
                self.boundary_points = []
                self.boundary_state = "recording"
            return {"boundary_state": "recording", "points": len(self.boundary_points)}
        if action == "pause":
            if self.boundary_state == "recording":
                self.boundary_state = "paused"
            return {"boundary_state": self.boundary_state, "points": len(self.boundary_points)}
        if action == "resume":
            if self.boundary_state == "paused":
                self.boundary_state = "recording"
            return {"boundary_state": self.boundary_state, "points": len(self.boundary_points)}
        if action == "close":
            if len(self.boundary_points) < 3:
                return {"error": f"boundary needs at least 3 walked points, has {len(self.boundary_points)}"}
            xz = [(p[0], p[2]) for p in self.boundary_points]
            simplified = simplify_closed_loop(xz, 0.3)
            if len(simplified) < 3:
                return {"error": "walked loop simplifies to fewer than 3 corners"}
            y = self.boundary_points[0][1]
            self.boundary_polygon = [[x, y, z] for x, z in simplified]
            self.boundary_state = "closed"
            report = self.boundary_report()
            return {"boundary_state": "closed", **report}
        return {"error": f"unknown boundary action {action!r} (use start, pause, resume or close)"}

    def _boundary_xz(self) -> list[tuple[float, float]] | None:
        if not self.boundary_polygon:
            return None
        return [(v[0], v[2]) for v in self.boundary_polygon]

    def boundary_report(self) -> dict:
        polygon = self._boundary_xz()
        if polygon is None:
            return {"boundary_state": self.boundary_state}
        area = 0.0
        perimeter = 0.0
        for i in range(len(polygon)):
            x1, z1 = polygon[i]
            x2, z2 = polygon[(i + 1) % len(polygon)]
            area += x1 * z2 - x2 * z1
            perimeter += math.hypot(x2 - x1, z2 - z1)
        inside, edge = [], []
        for live_id in self.live_ids():
            snap = self.snapshots.get(live_id) or {}
            position = snap.get("position_m")
            if not position:
                continue
            if point_in_polygon(position[0], position[2], polygon):
                inside.append(live_id)
            else:
                dist = min(
                    _point_segment_distance(
                        position[0], position[2],
                        polygon[i][0], polygon[i][1],
                        polygon[(i + 1) % len(polygon)][0], polygon[(i + 1) % len(polygon)][1],
                    )
                    for i in range(len(polygon))
                )
                if dist <= 2.0:
                    edge.append(live_id)
        return {
            "boundary_state": "closed",
            "area_m2": round(abs(area) / 2.0, 1),
            "perimeter_m": round(perimeter, 1),
            "corners": len(polygon),
            "vertices_m": self.boundary_polygon,
            "trees_inside": [tree_label(i) for i in inside],
            "trees_on_edge": [tree_label(i) for i in edge],
            "confirmed_inside": sum(
                1 for i in inside if (self.snapshots.get(i) or {}).get("status") == "confirmed"
            ),
        }

    # -- reports ---------------------------------------------------------------------
    def _scoped_ids(self) -> tuple[list[int], str]:
        polygon = self._boundary_xz()
        if polygon is None:
            return self.live_ids(), "whole walk (no boundary closed)"
        inside = []
        for live_id in self.live_ids():
            position = (self.snapshots.get(live_id) or {}).get("position_m")
            if position and point_in_polygon(position[0], position[2], polygon):
                inside.append(live_id)
        return inside, "closed boundary"

    def survey_totals(self) -> dict:
        """Whole-walk logged totals, whatever the boundary scope is.

        One definition, used by the packet's LOGGED line and pinned by
        test: live trees with a position, split by the snapshot's own
        confirmed flag; complete is the usual completeness check among
        the confirmed; with_species counts a stored species result.
        """
        ids = [
            i
            for i in self.live_ids()
            if (self.snapshots.get(i) or {}).get("position_m")
        ]
        confirmed = [
            i for i in ids if (self.snapshots.get(i) or {}).get("status") == "confirmed"
        ]
        return {
            "trees_total": len(ids),
            "trees_confirmed": len(confirmed),
            "trees_not_yet_confirmed": len(ids) - len(confirmed),
            "confirmed_complete": sum(1 for i in confirmed if self.is_complete(i)),
            "with_species": sum(1 for i in ids if self.species.get(i)),
        }

    def area_report(self) -> dict:
        ids, scope = self._scoped_ids()
        # A tree with no position yet is a detection, not a placed tree in
        # the area; it does not enter the area counts.
        ids = [i for i in ids if (self.snapshots.get(i) or {}).get("position_m")]
        confirmed = [i for i in ids if (self.snapshots.get(i) or {}).get("status") == "confirmed"]
        tentative = [i for i in ids if (self.snapshots.get(i) or {}).get("status") != "confirmed"]
        needing = {kind: 0 for kind in ("unconfirmed", "no_species")}
        for live_id in confirmed:
            for kind in set(self.needs(live_id)) & set(needing):
                needing[kind] += 1
        incomplete = []
        for live_id in confirmed:
            if not self.is_complete(live_id):
                row = self.tree_row(live_id)
                if row is not None:
                    incomplete.append(row)
        incomplete.sort(key=lambda r: (r["dist_m"] is None, r["dist_m"] or 0.0))
        report: dict = {
            "scope": scope,
            "trees_confirmed": len(confirmed),
            "confirmed_complete": sum(1 for i in confirmed if self.is_complete(i)),
            "trees_tentative": len(tentative),
            "confirmed_needing": needing,
            "nearest_to_fix": [
                {"id": r["id"], "dist_m": r["dist_m"], "clock": r["clock"], "needs": r["needs"]}
                for r in incomplete[:3]
            ],
            "coverage": round(self.coverage_fraction(), 2),
        }
        unseen = self.nearest_unseen_ground()
        if unseen is not None and report["coverage"] < 1.0:
            report["nearest_unseen_ground"] = {"dist_m": unseen["dist_m"], "clock": unseen["clock"]}
        if self.boundary_polygon:
            report.update(self.boundary_report())
        return report

    def route_to(self, live_id: int) -> dict:
        row = self.tree_row(live_id)
        if row is None:
            return {"error": f"no tree {tree_label(live_id)}"}
        result: dict = {
            "id": row["id"], "dist_m": row["dist_m"], "clock": row["clock"], "where": row["where"],
        }
        return result

    # -- packet ---------------------------------------------------------------------
    def packet(self, *, t_mono_ms: float = 0.0, t_wall_ms=None) -> str:
        pose = self.current_pose
        lines = []
        if pose is None:
            lines.append("NOW: no pose yet; tracking unknown.")
        else:
            pos = pose["position_m"]
            if t_wall_ms:
                from datetime import UTC, datetime

                when = datetime.fromtimestamp(t_wall_ms / 1000.0, tz=UTC).strftime("%H:%M:%S")
            else:
                when = f"{t_mono_ms / 1000.0:.0f}s into the walk"
            pitch = self.camera_pitch_deg()
            lines.append(
                f"NOW: {when}; phone position (site-local m) "
                f"[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]; tracking {pose['tracking']} "
                f"(tracking losses this walk: {self.tracking_losses}); "
                f"walking speed {self.walking_speed_mps():.1f} m/s over the last 5 s; "
                f"camera pitch {pitch:.0f} deg; boundary {self.boundary_state}."
            )
        focus = self.focus_id()
        if focus is None:
            lines.append("FOCUS: none within 15 deg and 8 m.")
        else:
            lines.append(f"FOCUS: {format_tree_row(self.tree_row(focus))}")
        rows = self.scene_rows(12.0)
        confirmed = [r for r in rows if r["status"] == "confirmed"][:6]
        tentative = [r for r in rows if r["status"] != "confirmed" and r["views"] >= 3][:2]
        extra = sum(
            1 for r in rows if r["status"] != "confirmed" and r["views"] >= 3
        ) - len(tentative)
        junk = sum(1 for r in rows if r["status"] != "confirmed" and r["views"] < 3)
        lines.append("AROUND YOU within 12 m:")
        shown = confirmed + tentative
        if shown:
            lines.extend(f"- {format_tree_row(r)}" for r in shown)
        else:
            lines.append("- none")
        more = extra + junk
        if more > 0:
            lines.append(f"+{more} more tentative detections")
        # Answers to the common questions, precomputed so a turn about
        # them needs no tool round: nearest outright, nearest work, and
        # one whole-walk totals line that means the same on every turn.
        nearest = sorted(
            (r for r in self.scene_rows(1e6) if r["dist_m"] is not None),
            key=lambda r: r["dist_m"],
        )[:3]
        if nearest:
            lines.append(
                "NEAREST (any status): "
                + "; ".join(
                    f"{r['id']} {r['dist_m']:.0f} m at {r['clock']} o'clock"
                    + ("" if r["status"] == "confirmed" else " (not confirmed yet)")
                    for r in nearest
                )
                + "."
            )
        report = self.area_report()
        needing = report["confirmed_needing"]
        fixable = [f for f in report.get("nearest_to_fix") or [] if f["dist_m"] is not None]
        if fixable:
            lines.append(
                "NEEDS WORK, nearest first: "
                + "; ".join(
                    f"{f['id']} ({', '.join(f['needs'])}) {f['dist_m']:.0f} m at {f['clock']} o'clock"
                    for f in fixable[:2]
                )
                + "."
            )
        unseen = report.get("nearest_unseen_ground")
        unseen_text = (
            f"; nearest unseen ground {unseen['dist_m']:.0f} m at {unseen['clock']} o'clock"
            if unseen else ""
        )
        lines.append(
            f"SURVEY [{report['scope']}]: {report['trees_confirmed']} confirmed "
            f"({report['confirmed_complete']} complete), "
            f"{report['trees_tentative']} not yet confirmed; confirmed needs: "
            f"no_species {needing['no_species']}; "
            f"coverage {report['coverage']:.0%}{unseen_text}."
        )
        totals = self.survey_totals()
        lines.append(
            f"LOGGED (whole walk, use these for 'how many have I logged'): "
            f"{totals['trees_confirmed']} confirmed, "
            f"{totals['trees_not_yet_confirmed']} not yet confirmed, "
            f"{totals['trees_total']} in total, "
            f"{totals['confirmed_complete']} complete, "
            f"{totals['with_species']} with a species result."
        )
        return "\n".join(lines)


def format_tree_row(row: dict | None) -> str:
    if row is None:
        return "none"
    parts = [row["id"], str(row["status"])]
    if row.get("dist_m") is not None:
        parts.append(f"{row['dist_m']:.0f} m at {row['clock']} o'clock ({row['where']})")
    if row.get("sigma_m") is not None:
        parts.append(f"position +/-{float(row['sigma_m']):.2f} m")
    else:
        parts.append("position unknown accuracy")
    parts.append(f"{row['views']} views")
    species = row.get("species")
    if species and species.get("taxon"):
        conf = species.get("confidence")
        conf_text = f" {float(conf):.0%}" if conf is not None else ""
        genus = f" (genus {species['taxon']})" if species.get("rank") == "genus" else ""
        runner = f", runner-up {species['runner_up']}" if species.get("runner_up") else ""
        parts.append(f"species {species['taxon']}{conf_text}{genus}{runner}")
    if row.get("needs"):
        parts.append(f"needs: {', '.join(row['needs'])}")
    if row.get("evidence_needs"):
        asks = ", ".join(n["want"] for n in row["evidence_needs"])
        parts.append(f"evidence next: {asks}")
    if row.get("notes"):
        parts.append(f"notes: {'; '.join(row['notes'])}")
    return "; ".join(parts)
