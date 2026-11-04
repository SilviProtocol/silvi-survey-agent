"""People correcting records after the fact, kept apart from model output.

A correction is a small dict appended to a list that lives next to the
canonical records, never inside them: the observation schema has no room
for an audit trail, and the model's original answer must stay readable
exactly as it was produced. Each entry names the record it talks about,
what the person said should change, who said it, whether it arrived by
voice or by tap, and when. Applying one may touch ordinary record fields
(a planted flag, two fresh child records after a split), but the entry
itself is history and is never rewritten or removed.

Record references use whichever identifier the speaker had to hand: a
survey local id such as ``tree-3``, or ``plant-<id>`` for a walk plant.
"""

from __future__ import annotations

import json
from pathlib import Path

OPERATIONS = ("exclude", "split", "merge_hint", "mark_planted", "set_species", "note")

_SOURCES = ("voice", "tap")
_WHO = ("walker", "reviewer")


def make_correction(
    correction_id: str,
    *,
    record_ref: str,
    operation: str,
    value: str | None = None,
    other_ref: str | None = None,
    who: str = "walker",
    source: str = "voice",
    at: str,
    transcript: str | None = None,
) -> dict:
    """Build one validated correction entry."""
    if operation not in OPERATIONS:
        raise ValueError(f"unknown correction operation: {operation!r}")
    if source not in _SOURCES:
        raise ValueError(f"correction source must be one of {_SOURCES}")
    if who not in _WHO:
        raise ValueError(f"correction who must be one of {_WHO}")
    if operation in ("set_species", "note") and not (value or "").strip():
        raise ValueError(f"{operation} needs the words the person used as its value")
    if operation == "merge_hint" and not (other_ref or value):
        raise ValueError("merge_hint needs the other record it points at")
    entry = {
        "correction_id": correction_id,
        "record_ref": str(record_ref),
        "operation": operation,
        "value": value,
        "other_ref": other_ref or (value if operation == "merge_hint" else None),
        "who": who,
        "source": source,
        "at": at,
        "transcript": transcript,
    }
    if operation == "merge_hint":
        entry["value"] = None
    return entry


def flags_by_ref(corrections) -> dict[str, dict]:
    """Fold the correction list into per-record display/export flags.

    Later entries win where they contradict earlier ones (a second human
    species name replaces the first as the export name, while both stay
    in the timeline). Split children and both ends of a merge hint are
    flagged for review; excluded records are flagged, not deleted.
    """
    flags: dict[str, dict] = {}

    def slot(ref: str) -> dict:
        return flags.setdefault(
            str(ref),
            {
                "excluded": False,
                "needs_review": False,
                "split_into": [],
                "split_from": None,
                "merge_hint_with": [],
                "planted_by": None,
                "human_species": None,
                "notes": [],
                "timeline": [],
            },
        )

    for entry in corrections or ():
        ref = str(entry["record_ref"])
        mine = slot(ref)
        mine["timeline"].append(entry)
        op = entry["operation"]
        if op == "exclude":
            mine["excluded"] = True
        elif op == "split":
            children = [str(c) for c in entry.get("child_refs") or ()]
            mine["split_into"] = children
            mine["needs_review"] = True
            for child in children:
                theirs = slot(child)
                theirs["needs_review"] = True
                theirs["split_from"] = ref
        elif op == "merge_hint" and entry.get("other_ref"):
            other = str(entry["other_ref"])
            if other not in mine["merge_hint_with"]:
                mine["merge_hint_with"].append(other)
            theirs = slot(other)
            if ref not in theirs["merge_hint_with"]:
                theirs["merge_hint_with"].append(ref)
            mine["needs_review"] = True
            theirs["needs_review"] = True
        elif op == "mark_planted":
            mine["planted_by"] = {"who": entry["who"], "at": entry["at"]}
        elif op == "set_species":
            mine["human_species"] = {
                "name": entry["value"],
                "who": entry["who"],
                "at": entry["at"],
                "source": entry["source"],
            }
        elif op == "note":
            mine["notes"].append(entry["value"])
    return flags


def corrections_for(corrections, *refs: str) -> list[dict]:
    """Entries that talk about any of the given record references."""
    wanted = {str(r) for r in refs if r}
    return [e for e in corrections or () if str(e["record_ref"]) in wanted]


def group_by_record_id(records, corrections) -> dict[str, list[dict]]:
    """Map canonical records (by local id) to their corrections, for export."""
    grouped: dict[str, list[dict]] = {}
    for record in records:
        local_id = record["identity"]["local_id"]
        mine = corrections_for(corrections, local_id)
        if mine:
            grouped[local_id] = mine
    return grouped


def load_corrections(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = data["corrections"] if isinstance(data, dict) else data
    out = []
    for i, raw in enumerate(entries):
        entry = make_correction(
            raw.get("correction_id") or f"corr-{i + 1}",
            record_ref=raw["record_ref"],
            operation=raw["operation"],
            value=raw.get("value"),
            other_ref=raw.get("other_ref"),
            who=raw.get("who") or "walker",
            source=raw.get("source") or "tap",
            at=raw.get("at") or "",
            transcript=raw.get("transcript"),
        )
        if raw.get("child_refs"):
            entry["child_refs"] = list(raw["child_refs"])
        out.append(entry)
    return out


def dump_corrections(corrections) -> dict:
    return {"corrections": list(corrections or ())}
