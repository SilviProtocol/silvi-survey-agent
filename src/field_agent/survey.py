"""Survey state: sites, boundaries, plots, plantings, observations.

The five create-tools land here. Plantings and observations become canonical
observation records, each validated against the record schema;
sites, boundaries and plots are survey entities the record points at.
Crosshair ground hits accumulate as corners until a polygon is closed.
Everything here is plain code: voice and tap both end at these methods.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from field_agent.contracts import validate_observation


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or "site"


def polygon_area_m2(vertices_m) -> float:
    """Shoelace area in the XZ ground plane."""
    pts = [(float(v[0]), float(v[2])) for v in vertices_m]
    total = 0.0
    for a in range(len(pts)):
        x1, z1 = pts[a]
        x2, z2 = pts[(a + 1) % len(pts)]
        total += x1 * z2 - x2 * z1
    return abs(total) / 2.0


@dataclass
class SurveyState:
    walk_id: str = "session"
    site: dict | None = None
    boundaries: list[dict] = field(default_factory=list)
    plots: list[dict] = field(default_factory=list)
    records: list[dict] = field(default_factory=list)
    # People's corrections, appended in arrival order and never edited.
    # The canonical record schema carries no audit trail, so the trail
    # lives here beside the records it talks about.
    corrections: list[dict] = field(default_factory=list)
    corners: list[list[float]] = field(default_factory=list)
    _seq: int = 0
    _correction_seq: int = 0

    def _next(self) -> int:
        self._seq += 1
        return self._seq

    @property
    def site_id(self) -> str:
        return self.site["site_id"] if self.site else "unsited"

    # -- entities ----------------------------------------------------------------
    def create_site(self, name: str) -> dict:
        self.site = {"site_id": _slug(name), "name": name}
        return dict(self.site)

    def _add_polygon(self, kind: str, label: str, vertices_m) -> dict:
        vertices = [[float(c) for c in v] for v in vertices_m]
        if len(vertices) < 3:
            raise ValueError(f"{kind} needs at least three vertices")
        entity = {
            "id": f"{kind}-{self._next()}",
            "kind": kind,
            "label": label,
            "site_id": self.site_id,
            "vertices_m": vertices,
            "area_m2": round(polygon_area_m2(vertices), 3),
        }
        (self.boundaries if kind == "boundary" else self.plots).append(entity)
        return entity

    def create_boundary(self, label: str, vertices_m) -> dict:
        return self._add_polygon("boundary", label, vertices_m)

    def create_plot(self, label: str, vertices_m) -> dict:
        return self._add_polygon("plot", label, vertices_m)

    # -- crosshair -----------------------------------------------------------------
    def add_corner(self, position_m) -> int:
        """Add one crosshair ground hit. Returns the corner count."""
        self.corners.append([float(c) for c in position_m])
        return len(self.corners)

    def close_polygon(self, label: str, kind: str = "plot") -> dict:
        if kind not in ("boundary", "plot"):
            raise ValueError("polygon kind must be boundary or plot")
        if len(self.corners) < 3:
            raise ValueError(f"need at least 3 corners to close a polygon, have {len(self.corners)}")
        entity = self._add_polygon(kind, label, list(self.corners))
        self.corners = []
        return entity

    # -- records -----------------------------------------------------------------
    def _record(
        self,
        *,
        label: str,
        position_m,
        sigma_m,
        establishment: str,
        observed_at: str,
        taxon: str | None = None,
        observer: str | None = None,
    ) -> dict:
        n = self._next()
        local_id = f"tree-{n}"
        site_id = self.site_id
        record = {
            "id": f"urn:silvi:obs:{site_id}:{local_id}:v1",
            "site_id": site_id,
            "identity": {
                "local_id": local_id,
                "persistent_id": f"urn:silvi:plant:{site_id}:{n}",
                "status": "tentative",
                "merged_into": None,
                "visit_ids": [self.walk_id],
            },
            "position": {
                "frame": "site_local",
                "xyz_m": [float(c) for c in position_m] if position_m is not None else None,
                "geo": None,
            },
            "uncertainty": {
                "sigma_m": float(sigma_m) if sigma_m is not None else None,
                "method": "field_placement" if position_m is not None else "bearing_only",
                "n_views": 1,
                "status": "placed" if position_m is not None else "bearing_only",
            },
            "collection": {
                "walk_id": self.walk_id,
                "observed_at": observed_at,
                "observer": observer,
                "device": None,
                "protocol": "field-agent-survey-v1",
            },
            "identification": (
                {
                    "taxon": taxon,
                    "confidence": 0.5,
                    "basis": "walker_stated",
                    "identified_by": observer,
                    "revisable": True,
                }
                if taxon
                else None
            ),
            "verbatim_label": label,
            "establishment": establishment,
            "measurements": {"height_m": None, "dbh_cm": None},
            "media": [],
        }
        validate_observation(record)
        self.records.append(record)
        return record

    def create_planting(
        self,
        position_m,
        label: str,
        *,
        observed_at: str,
        taxon: str | None = None,
        sigma_m: float | None = None,
        observer: str | None = None,
    ) -> dict:
        return self._record(
            label=label,
            position_m=position_m,
            sigma_m=sigma_m,
            establishment="planted",
            observed_at=observed_at,
            taxon=taxon,
            observer=observer,
        )

    def create_observation(
        self,
        position_m,
        label: str,
        *,
        observed_at: str,
        sigma_m: float | None = None,
        establishment: str = "unknown",
        observer: str | None = None,
    ) -> dict:
        return self._record(
            label=label,
            position_m=position_m,
            sigma_m=sigma_m,
            establishment=establishment,
            observed_at=observed_at,
            observer=observer,
        )

    def attach_identification(self, local_id: str, identification: dict) -> dict:
        """Attach (or revise) a record's RI-style species identification."""
        record = self.find_record(local_id)
        if record is None:
            raise ValueError(f"no stored record for {local_id!r}")
        record["identification"] = identification
        if self.walk_id not in record["identity"]["visit_ids"]:
            record["identity"]["visit_ids"].append(self.walk_id)
        validate_observation(record)
        return record

    def record_growth(
        self,
        local_id: str,
        *,
        height_m: float | None = None,
        dbh_cm: float | None = None,
    ) -> dict:
        """Set growth measurements on an existing record; add this visit."""
        record = self.find_record(local_id)
        if record is None:
            raise ValueError(f"no stored record for {local_id!r}")
        if height_m is not None:
            record["measurements"]["height_m"] = float(height_m)
        if dbh_cm is not None:
            record["measurements"]["dbh_cm"] = float(dbh_cm)
        if self.walk_id not in record["identity"]["visit_ids"]:
            record["identity"]["visit_ids"].append(self.walk_id)
        validate_observation(record)
        return record

    # -- people's corrections ------------------------------------------------------
    def next_correction_id(self) -> str:
        self._correction_seq += 1
        return f"corr-{self._correction_seq}"

    def apply_correction(self, entry: dict) -> dict:
        """Append one correction and land the part that changes records.

        The entry itself is history and is stored as given. Only three
        operations touch canonical records, and only through fields the
        schema already has: ``mark_planted`` sets establishment, and
        ``split`` mints two child records at the parent's position (the
        person said there are two plants, not where each one stands, so
        no position is invented — both children carry the parent's spot
        until somebody refines them, flagged for review through the
        correction flags). Excluding, naming a species, merge hints and
        notes change no record fields at all; views and exports read
        them from the correction list, so model output stays untouched.
        """
        record = self.find_record(str(entry["record_ref"]))
        if record is None:
            raise ValueError(f"no stored record for {entry['record_ref']!r}")
        stored = dict(entry)
        op = stored["operation"]
        if op == "mark_planted":
            record["establishment"] = "planted"
            validate_observation(record)
        elif op == "split":
            children = []
            for part in ("one of a split pair", "two of a split pair"):
                child = self._record(
                    label=f"{record.get('verbatim_label') or 'plant'} ({part})",
                    position_m=record["position"]["xyz_m"],
                    sigma_m=record["uncertainty"]["sigma_m"],
                    establishment=record["establishment"],
                    observed_at=stored["at"],
                    observer=stored["who"],
                )
                children.append(child["identity"]["local_id"])
            stored["child_refs"] = children
        elif op == "merge_hint":
            other = self.find_record(str(stored.get("other_ref") or ""))
            if other is None:
                raise ValueError(
                    f"no stored record for merge partner {stored.get('other_ref')!r}"
                )
            stored["other_ref"] = other["identity"]["local_id"]
        self.corrections.append(stored)
        return stored

    # -- promotion from walk plant snapshots --------------------------------
    def promote_from_snapshots(self, snapshots, *, match_radius_m: float = 1.0) -> dict:
        """Promote/merge mid-walk records backed by walk plant snapshots.

        A snapshot is a wire ``plant_snapshot`` payload (plant_id, status,
        position_m, sigma_m, n_views, merged_into). Rules, in the record
        identity fields (``status`` / ``merged_into`` / ``visit_ids``):

        - A record matches its nearest snapshot within
``match_radius_m`` on the XZ plane.
        - Backed by a **confirmed** snapshot -> record becomes confirmed;
          sigma/n_views adopt the snapshot's when present.
        - Two records backed by the same confirmed snapshot are duplicates:
          the earliest survives (confirmed, visit_ids unioned), the rest
          become ``merged`` with ``merged_into`` = survivor persistent_id.
        - A record matching a **merged** snapshot merges into the record
          backed by that snapshot's ``merged_into`` target, when one exists.
        - Anything else (no match, tentative/retired snapshot) is unbacked
          and stays tentative.

        Already-confirmed/merged records are left alone (idempotent).
        Returns a report dict; ``backed_by`` maps local_id -> plant_id.
        """
        def xz(pos):
            return (float(pos[0]), float(pos[2])) if pos else None

        snaps = []
        for snap in snapshots:
            pos = xz(snap.get("position_m"))
            if pos is not None:
                snaps.append((int(snap["plant_id"]), snap, pos))
        by_plant = {pid: snap for pid, snap, _pos in snaps}

        def nearest(record):
            pos = xz(record["position"]["xyz_m"])
            if pos is None:
                return None
            best = None
            for pid, snap, spos in snaps:
                dist = math.hypot(pos[0] - spos[0], pos[1] - spos[1])
                if dist <= match_radius_m and (
                    best is None or dist < best[0]
                ):
                    best = (dist, pid, snap)
            return best

        report: dict = {"promoted": [], "merged": {}, "unbacked": [], "backed_by": {}}
        matched: dict[str, tuple[int, dict]] = {}
        for record in self.records:
            identity = record["identity"]
            if identity["status"] in ("confirmed", "merged"):
                continue
            found = nearest(record)
            if found is None:
                report["unbacked"].append(identity["local_id"])
            else:
                _dist, pid, snap = found
                matched[identity["local_id"]] = (pid, snap)
                report["backed_by"][identity["local_id"]] = pid

        # Confirmed backing, grouping duplicates by backing plant.
        survivor_by_plant: dict[int, dict] = {}
        for record in self.records:
            local_id = record["identity"]["local_id"]
            if local_id not in matched:
                continue
            pid, snap = matched[local_id]
            status = snap.get("status")
            if status == "merged" and snap.get("merged_into") is not None:
                target = by_plant.get(int(snap["merged_into"]))
                if target is not None and target.get("status") == "confirmed":
                    pid, snap = int(snap["merged_into"]), target
                    status = "confirmed"
                    report["backed_by"][local_id] = pid
            if status != "confirmed":
                report["unbacked"].append(local_id)
                continue
            survivor = survivor_by_plant.get(pid)
            if survivor is None:
                survivor_by_plant[pid] = record
                identity = record["identity"]
                identity["status"] = "confirmed"
                if snap.get("sigma_m") is not None:
                    record["uncertainty"]["sigma_m"] = float(snap["sigma_m"])
                if snap.get("n_views"):
                    record["uncertainty"]["n_views"] = int(snap["n_views"])
                validate_observation(record)
                report["promoted"].append(local_id)
            else:
                identity = record["identity"]
                identity["status"] = "merged"
                identity["merged_into"] = survivor["identity"]["persistent_id"]
                for visit in identity["visit_ids"]:
                    if visit not in survivor["identity"]["visit_ids"]:
                        survivor["identity"]["visit_ids"].append(visit)
                validate_observation(record)
                validate_observation(survivor)
                report["merged"][local_id] = survivor["identity"]["local_id"]
        return report

    # -- lookup / summary -----------------------------------------------------------
    def find_record(self, local_id: str) -> dict | None:
        wanted = str(local_id).strip().lower()
        for record in self.records:
            lid = record["identity"]["local_id"]
            if wanted in (lid.lower(), f"plant-{lid.rsplit('-', 1)[-1]}", lid.rsplit("-", 1)[-1]):
                return record
        return None

    def summary(self) -> str:
        site = self.site["name"] if self.site else "no site yet"
        return (
            f"site: {site}; boundaries: {len(self.boundaries)}; plots: {len(self.plots)}; "
            f"records: {len(self.records)}; open corners: {len(self.corners)}"
        )
